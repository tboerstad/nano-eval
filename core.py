"""
Core utilities for nano-eval.

Responsibilities:
- ApiConfig: endpoint, model, concurrency, timeout
- Sample/Task: declarative task definition (dataset config + scorer)
- complete(): async batch chat completions (OpenAI-compatible)
- run_task(): evaluate a Task, return TaskResult
- _encode_image(): PIL→base64; rejects remote URLs
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import datasets
import datasets.config as ds_config
import httpx
from datasets import Dataset, DownloadMode
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("nano_eval.core")


class Metrics(TypedDict):
    accuracy: float
    accuracy_stderr: float


class ApiResponse(TypedDict):
    answer: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float


class RequestLogEntry(TypedDict):
    request_id: int
    target: str
    prompt: str
    response: str
    score: float
    stop_reason: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float


class TaskResult(TypedDict):
    elapsed_seconds: float
    metrics: Metrics
    num_samples: int
    samples_hash: str
    task: str
    modality: str
    total_input_tokens: int
    total_output_tokens: int
    tokens_per_second: float


@dataclass(frozen=True)
class Prompt:
    """Evaluation prompt with optional images for multimodal tasks."""

    text: str | list[dict[str, str]]
    images: list[Any] = field(default_factory=list)


@dataclass
class Sample:
    """A single evaluation sample: prompt + expected target."""

    prompt: Prompt
    target: str


@dataclass(frozen=True)
class Task:
    """Declarative task definition: dataset config + scoring function."""

    name: str
    dataset: str
    revision: str
    splits: list[str]
    extract: Callable[[dict[str, Any]], Sample]
    score: Callable[[str, str], float]
    config_name: str | None = None

    def load_samples(
        self, max_samples: int | None = None, seed: int | None = None
    ) -> list[Sample]:
        return load_hf_samples(
            dataset=self.dataset,
            revision=self.revision,
            splits=self.splits,
            extract=self.extract,
            max_samples=max_samples,
            seed=seed,
            config_name=self.config_name,
        )


@dataclass
class ApiConfig:
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
) -> ApiResponse:
    """Single request. Raises RuntimeError on failure."""
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload)
    duration = time.perf_counter() - t0
    if resp.is_success:
        data = resp.json()
        return ApiResponse(
            answer=data["choices"][0]["message"]["content"],
            stop_reason=data["choices"][0]["finish_reason"],
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            duration_seconds=duration,
        )
    raise RuntimeError(f"Request failed: {resp.text}")


async def complete(
    prompts: list[Prompt],
    config: ApiConfig,
    progress_desc: str = "Running evals",
) -> list[ApiResponse]:
    """
    Run batch of chat completions.

    Args:
        prompts: List of prompts (text-only or multimodal with images)
        config: API configuration (includes gen_kwargs for temperature, max_tokens, etc.)
        progress_desc: Label shown on the tqdm progress bar
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
        tasks: list[asyncio.Task[ApiResponse]] = []
        for prompt in prompts:
            if prompt.images:
                assert isinstance(prompt.text, str)
                messages = _build_vision_message(prompt.text, prompt.images)
            elif isinstance(prompt.text, list):
                messages = prompt.text
            else:
                messages = [{"role": "user", "content": prompt.text}]

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
            # On any failure (including Ctrl-C), cancel all pending tasks and await
            # them to properly "retrieve" their exceptions. Without this, Python logs
            # "Task exception was never retrieved" for each concurrent task that failed.
            # Using BaseException (not Exception) ensures cleanup runs on KeyboardInterrupt.
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
    content.append({"type": "text", "text": text.strip()})
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


def _prompt_to_str(prompt: Prompt) -> str:
    """Extract text from prompt."""
    if isinstance(prompt.text, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in prompt.text)
    return prompt.text


def _log_sample_results(
    samples: list[Sample],
    responses: list[ApiResponse],
    scores: list[float],
) -> None:
    """Log each sample's prompt, answer, and result at DEBUG level."""
    n = len(samples)
    for i, (s, r, score) in enumerate(zip(samples, responses, scores)):
        prompt_text = _prompt_to_str(s.prompt).replace("\\n", "\n")
        answer_text = r["answer"].replace("\\n", "\n")
        emoji = "✅" if score == 1.0 else "❌"
        logger.debug(
            f"\n{'=' * 60}\n"
            f"Sample {i + 1}/{n} {emoji}\n"
            f"{'=' * 60}\n"
            f"PROMPT:\n{prompt_text}\n"
            f"{'-' * 60}\n"
            f"ANSWER:\n{answer_text}\n"
            f"{'-' * 60}\n"
            f"TARGET: {s.target}\n"
            f"{'=' * 60}"
        )


def compute_samples_hash(samples: list[Sample]) -> str:
    """Compute SHA256 hash for all samples in a task (includes image data)."""
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(_prompt_to_str(s.prompt).encode())
        for img in s.prompt.images:
            hasher.update(_encode_image(img).encode())
        hasher.update(s.target.encode())
    return hasher.hexdigest()


async def run_task(
    task: Task,
    config: ApiConfig,
    modality: str,
    max_samples: int | None = None,
    seed: int | None = None,
) -> tuple[TaskResult, list[RequestLogEntry]]:
    """
    Evaluate a task: collect samples, run inference, compute scores.

    Args:
        task: Task definition with samples loader and scoring function
        config: API configuration (includes gen_kwargs for temperature, max_tokens, etc.)
        modality: Modality label for logging and result metadata
        max_samples: Optional limit on number of samples
        seed: Optional seed for shuffling sample order

    Returns:
        TaskResult with metrics, sample count, elapsed time, and per-sample data
    """
    samples = task.load_samples(max_samples, seed)
    samples_hash = compute_samples_hash(samples)
    prompts = [s.prompt for s in samples]

    logger.info(
        f"Starting {modality} ({task.name}) eval: "
        f"{len(samples)} samples, up to {config.max_concurrent} concurrent requests"
    )
    t0 = time.perf_counter()
    responses = await complete(prompts, config, f"Running {modality} eval")
    elapsed = time.perf_counter() - t0

    # Log warning if any responses did not complete with "stop"
    reason_counts = Counter(r["stop_reason"] for r in responses)
    non_stop_count = sum(c for reason, c in reason_counts.items() if reason != "stop")
    if non_stop_count:
        logger.warning(
            f"{non_stop_count} response(s) did not finish with 'stop'. "
            f"Reasons: {dict(reason_counts)}"
        )

    scores = [task.score(r["answer"], s.target) for r, s in zip(responses, samples)]
    n = len(samples)

    if logger.isEnabledFor(logging.DEBUG):
        _log_sample_results(samples, responses, scores)

    accuracy = sum(scores) / n if n else 0.0
    stderr = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0

    logger.debug(f"{task.name}: accuracy={accuracy:.4f}±{stderr:.4f} ({elapsed:.2f}s)")

    # Collect per-request data for optional JSONL export (negligible overhead)
    request_logs: list[RequestLogEntry] = [
        RequestLogEntry(
            request_id=i,
            target=s.target,
            prompt=_prompt_to_str(s.prompt),
            response=r["answer"],
            score=score,
            stop_reason=r["stop_reason"],
            input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"],
            duration_seconds=r["duration_seconds"],
        )
        for i, (s, r, score) in enumerate(zip(samples, responses, scores))
    ]
    total_input_tokens = sum(r["input_tokens"] for r in responses)
    total_output_tokens = sum(r["output_tokens"] for r in responses)
    total_tokens = total_input_tokens + total_output_tokens
    total_duration = sum(r["duration_seconds"] for r in responses)
    result = TaskResult(
        elapsed_seconds=elapsed,
        metrics=Metrics(accuracy=accuracy, accuracy_stderr=stderr),
        num_samples=n,
        samples_hash=samples_hash,
        task=task.name,
        modality=modality,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        tokens_per_second=total_tokens / total_duration if total_duration else 0.0,
    )
    return result, request_logs


def load_hf_samples(
    dataset: str,
    revision: str,
    splits: list[str],
    extract: Callable[[dict[str, Any]], Sample],
    max_samples: int | None,
    seed: int | None,
    config_name: str | None = None,
) -> list[Sample]:
    """Load samples from a HuggingFace dataset across splits."""
    # TODO Upstream fix. HF datasets logging is too noisy
    datasets.utils.logging.set_verbosity_error()

    # Enable offline mode if dataset is already cached (avoids HEAD requests)
    cache_path = (
        Path(HF_HUB_CACHE)
        / f"datasets--{dataset.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    cached = cache_path.is_dir()
    logger.info(f"Cache {'hit' if cached else 'miss'} for {dataset}, HF_HOME={HF_HOME}")
    old_offline = ds_config.HF_HUB_OFFLINE
    if cached:
        ds_config.HF_HUB_OFFLINE = True

    try:
        result: list[Sample] = []
        for split in splits:
            remaining = None if max_samples is None else max_samples - len(result)
            if remaining is not None and remaining <= 0:
                break
            ds = datasets.load_dataset(
                dataset,
                name=config_name,
                split=split,
                revision=revision,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
            assert isinstance(ds, Dataset)
            if seed is not None:
                ds = ds.shuffle(seed=seed)
            if remaining is not None:
                ds = ds.select(range(min(remaining, len(ds))))
            for doc in ds:
                result.append(extract(doc))
        return result
    finally:
        ds_config.HF_HUB_OFFLINE = old_offline
