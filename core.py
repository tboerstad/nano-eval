"""
Core utilities for nano-eval.

Responsibilities:
- APIConfig: endpoint, model, concurrency, timeout
- Sample/Task: minimal task abstraction (generator + scorer)
- complete(): async batch chat completions (OpenAI-compatible)
- run_task(): evaluate a Task, return TaskResult
- _normalize(): text normalization for comparison
- _encode_image(): PIL→base64; rejects remote URLs
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import datasets.config as ds_config
import httpx
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class Metrics(TypedDict):
    exact_match: float
    exact_match_stderr: float


class LoggedSample(TypedDict):
    sample_id: int
    target: str
    prompt: str
    response: str
    exact_match: float


class TaskResult(TypedDict):
    elapsed_seconds: float
    metrics: Metrics
    num_samples: int
    samples: NotRequired[list[LoggedSample]]
    samples_hash: str
    task: str
    task_type: str


@dataclass
class Sample:
    """A single evaluation sample: prompt + expected target."""

    prompt: (
        str | tuple[str, list[Any]] | list[dict[str, str]]
    )  # text, (text, images), or messages
    target: str


@dataclass(frozen=True)
class Task:
    """Minimal task definition: a loader of samples + a scoring function."""

    name: str
    task_type: str  # "text" or "vision"
    samples: Callable[[int | None, int | None], list[Sample]]  # (max_samples, seed)
    score: Callable[[str, str], float]  # (response, target) -> score


@dataclass
class APIConfig:
    """API configuration."""

    url: str
    model: str
    api_key: str = ""
    max_concurrent: int = 8
    timeout: int = 300
    gen_kwargs: dict[str, Any] = field(default_factory=dict)


async def _request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> str:
    """Single request. Raises RuntimeError on failure."""
    resp = await client.post(url, json=payload)
    if resp.is_success:
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Request failed: {resp.text}")


async def complete(
    prompts: list[str | tuple[str, list] | list[dict[str, str]]],
    config: APIConfig,
    progress_desc: str = "Running evals",
) -> list[str]:
    """
    Run batch of chat completions.

    Args:
        prompts: List of prompts. Each is either:
            - str: text-only prompt
            - tuple[str, list]: (text, images) for multimodal
            - list[dict]: pre-built messages for multiturn
        config: API configuration (includes gen_kwargs for temperature, max_tokens, etc.)

    Returns:
        List of response strings
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.max_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
        tasks: list[asyncio.Task[str]] = []
        for prompt in prompts:
            if isinstance(prompt, list):
                messages = prompt
            elif isinstance(prompt, tuple):
                text, images = prompt
                messages = _build_vision_message(text, images)
            else:
                messages = [{"role": "user", "content": prompt}]

            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                **config.gen_kwargs,
            }

            tasks.append(asyncio.create_task(_request(client, config.url, payload)))

        try:
            return list(
                await tqdm_asyncio.gather(*tasks, desc=progress_desc, leave=False)
            )
        except BaseException:
            # Cancel pending tasks and await them to suppress
            # "Task exception was never retrieved" warnings
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def _build_vision_message(text: str, images: list[Any]) -> list[dict[str, Any]]:
    """Build OpenAI vision API message."""
    content: list[dict[str, Any]] = []
    for img in images:
        if b64 := _encode_image(img):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    content.append({"type": "text", "text": text.replace("<image>", "").strip()})
    return [{"role": "user", "content": content}]


def _encode_image(image: Any) -> str:
    """Encode PIL image to base64, or pass through string."""
    if isinstance(image, str):
        if image.startswith("http"):
            raise ValueError("Remote image URLs are not supported.")
        return image

    if isinstance(image, Image.Image):
        try:
            # Convert to RGB if needed to avoid save errors with CMYK/palette modes
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buf = BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}") from e

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    return text.lower().strip()


def _prompt_to_str(prompt: str | tuple[str, list] | list[dict[str, str]]) -> str:
    """Extract text from prompt (handles multimodal tuples and message lists)."""
    if isinstance(prompt, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in prompt)
    return prompt[0] if isinstance(prompt, tuple) else prompt


def compute_samples_hash(samples: list[Sample]) -> str:
    """Compute SHA256 hash for all samples in a task (includes image data)."""
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(_prompt_to_str(s.prompt).encode())
        if isinstance(s.prompt, tuple):
            for img in s.prompt[1]:
                hasher.update(_encode_image(img).encode())
        hasher.update(s.target.encode())
    return hasher.hexdigest()


async def run_task(
    task: Task,
    config: APIConfig,
    max_samples: int | None = None,
    seed: int | None = None,
) -> TaskResult:
    """
    Evaluate a task: collect samples, run inference, compute scores.

    Args:
        task: Task definition with samples loader and scoring function
        config: API configuration (includes gen_kwargs for temperature, max_tokens, etc.)
        max_samples: Optional limit on number of samples
        seed: Optional seed for shuffling sample order

    Returns:
        TaskResult with metrics, sample count, elapsed time, and per-sample data
    """
    samples = task.samples(max_samples, seed)
    samples_hash = compute_samples_hash(samples)
    prompts = [s.prompt for s in samples]

    logger.info(
        f"Starting {task.task_type} ({task.name}) eval: "
        f"{len(samples)} samples, up to {config.max_concurrent} concurrent requests"
    )
    t0 = time.perf_counter()
    desc = "Running vision eval" if task.task_type == "vision" else "Running text eval"
    responses = await complete(prompts, config, desc)
    elapsed = time.perf_counter() - t0

    scores = [task.score(r, s.target) for r, s in zip(responses, samples)]
    n = len(samples)
    accuracy = sum(scores) / n if n else 0.0
    stderr = math.sqrt(accuracy * (1 - accuracy) / (n - 1)) if n > 1 else 0.0

    logger.debug(f"{task.name}: accuracy={accuracy:.4f}±{stderr:.4f} ({elapsed:.2f}s)")

    # Always collect per-sample data for optional JSONL export (negligible overhead)
    logged_samples: list[LoggedSample] = [
        LoggedSample(
            sample_id=i,
            target=s.target,
            prompt=_prompt_to_str(s.prompt),
            response=r,
            exact_match=score,
        )
        for i, (s, r, score) in enumerate(zip(samples, responses, scores))
    ]
    return TaskResult(
        elapsed_seconds=round(elapsed, 2),
        metrics=Metrics(exact_match=accuracy, exact_match_stderr=stderr),
        num_samples=n,
        samples=logged_samples,
        samples_hash=samples_hash,
        task=task.name,
        task_type=task.task_type,
    )


@contextmanager
def offline_if_cached(dataset: str, revision: str):
    """Context manager: enable HF offline mode if dataset is cached (avoids HEAD requests).

    Yields:
        Tuple of (cached: bool, cache_path: Path) where cached indicates if dataset
        is in cache and cache_path is the location of the cache directory.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = (
        hf_home
        / "hub"
        / f"datasets--{dataset.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    cached = cache.is_dir() and any(cache.iterdir())

    if cached:
        old = ds_config.HF_HUB_OFFLINE
        ds_config.HF_HUB_OFFLINE = True
    try:
        yield cached, cache
    finally:
        if cached:
            ds_config.HF_HUB_OFFLINE = old
