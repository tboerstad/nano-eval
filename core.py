"""
Core utilities for nano-eval.

Responsibilities:
- APIConfig: endpoint, model, concurrency, timeout
- Sample/Task: minimal task abstraction (generator + scorer)
- Input types: TextPrompt, VisionPrompt, SentencePair
- complete(): async batch chat completions (OpenAI-compatible)
- embed(): async batch embeddings
- run_task(): evaluate a Task, return TaskResult
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
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


# --- Input types ---


@dataclass(frozen=True)
class TextPrompt:
    """Simple text prompt for chat completion."""

    text: str


@dataclass(frozen=True)
class VisionPrompt:
    """Text prompt with images for vision models."""

    text: str
    images: list[Any]


@dataclass(frozen=True)
class SentencePair:
    """Two sentences for embedding similarity comparison."""

    first: str
    second: str


Input = TextPrompt | VisionPrompt | SentencePair | list[dict[str, str]]


# --- Result types ---


class Metrics(TypedDict):
    exact_match: float
    exact_match_stderr: float


class LoggedSample(TypedDict):
    sample_id: int
    target: str
    input: str
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


# --- Core abstractions ---


@dataclass
class Sample:
    """A single evaluation sample: input + expected target."""

    input: Input
    target: str


@dataclass(frozen=True)
class Task:
    """Minimal task definition: a loader of samples + a scoring function."""

    name: str
    task_type: str  # "text", "vision", or "embedding"
    samples: Callable[[int | None, int | None], list[Sample]]  # (max_samples, seed)
    score: Callable[[Any, str], float]  # (response, target) -> score


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
    inputs: list[Input],
    config: APIConfig,
    progress_desc: str = "Running evals",
) -> list[str]:
    """
    Run batch of chat completions.

    Args:
        inputs: List of inputs (TextPrompt, VisionPrompt, or messages list)
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
        for inp in inputs:
            if isinstance(inp, list):
                messages = inp
            elif isinstance(inp, VisionPrompt):
                messages = _build_vision_message(inp.text, inp.images)
            elif isinstance(inp, TextPrompt):
                messages = [{"role": "user", "content": inp.text}]
            elif isinstance(inp, SentencePair):
                raise ValueError("SentencePair should use embed(), not complete()")
            else:
                raise TypeError(f"Unknown input type: {type(inp)}")

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


async def _embed_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> list[list[float]]:
    """Single embedding request. Returns list of embedding vectors."""
    resp = await client.post(url, json=payload)
    if resp.is_success:
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
    raise RuntimeError(f"Embedding request failed: {resp.text}")


async def embed(
    texts: list[str],
    config: APIConfig,
    progress_desc: str = "Running embeddings",
) -> list[list[float]]:
    """
    Run batch of embedding requests.

    Args:
        texts: List of texts to embed
        config: API configuration

    Returns:
        List of embedding vectors (one per input text)
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    base_url = config.url.replace("/chat/completions", "")
    embed_url = f"{base_url}/embeddings"

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.max_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
        tasks: list[asyncio.Task[list[list[float]]]] = []
        for text in texts:
            payload: dict[str, Any] = {
                "model": config.model,
                "input": [text],
            }
            tasks.append(
                asyncio.create_task(_embed_request(client, embed_url, payload))
            )

        try:
            results = list(
                await tqdm_asyncio.gather(*tasks, desc=progress_desc, leave=False)
            )
            return [r[0] for r in results]
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def rank(data: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)
    d_squared = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1 - (6 * d_squared) / (n * (n**2 - 1))


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


def _input_to_str(inp: Input) -> str:
    """Convert input to string for logging."""
    if isinstance(inp, TextPrompt):
        return inp.text
    if isinstance(inp, VisionPrompt):
        return inp.text
    if isinstance(inp, SentencePair):
        return f"{inp.first} | {inp.second}"
    # Messages list
    return "\n".join(f"{m['role']}: {m['content']}" for m in inp)


def compute_samples_hash(samples: list[Sample]) -> str:
    """Compute SHA256 hash for all samples in a task (includes image data)."""
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(_input_to_str(s.input).encode())
        if isinstance(s.input, VisionPrompt):
            for img in s.input.images:
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
    n = len(samples)

    logger.info(
        f"Starting {task.task_type} ({task.name}) eval: "
        f"{n} samples, up to {config.max_concurrent} concurrent requests"
    )
    t0 = time.perf_counter()

    if task.task_type == "embedding":
        texts1 = []
        texts2 = []
        for s in samples:
            if isinstance(s.input, SentencePair):
                texts1.append(s.input.first)
                texts2.append(s.input.second)
            else:
                raise ValueError("Embedding task requires SentencePair inputs")

        all_texts = texts1 + texts2
        all_embeddings = await embed(all_texts, config, "Running embedding eval")
        embeddings1 = all_embeddings[:n]
        embeddings2 = all_embeddings[n:]

        pred_sims = [
            _cosine_similarity(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)
        ]
        gold_sims = [float(s.target) for s in samples]
        scores = pred_sims
        responses = [f"{sim:.4f}" for sim in pred_sims]

        spearman = _spearman_correlation(pred_sims, gold_sims)
        metric_value = spearman
        stderr = 0.0
        logger.debug(
            f"{task.name}: spearman={spearman:.4f} ({time.perf_counter() - t0:.2f}s)"
        )
    else:
        inputs = [s.input for s in samples]
        desc = (
            "Running vision eval" if task.task_type == "vision" else "Running text eval"
        )
        responses = await complete(inputs, config, desc)
        scores = [task.score(r, s.target) for r, s in zip(responses, samples)]
        metric_value = sum(scores) / n if n else 0.0
        stderr = (
            math.sqrt(metric_value * (1 - metric_value) / (n - 1)) if n > 1 else 0.0
        )
        logger.debug(
            f"{task.name}: accuracy={metric_value:.4f}±{stderr:.4f} ({time.perf_counter() - t0:.2f}s)"
        )

    elapsed = time.perf_counter() - t0

    logged_samples: list[LoggedSample] = [
        LoggedSample(
            sample_id=i,
            target=s.target,
            input=_input_to_str(s.input),
            response=r,
            exact_match=score,
        )
        for i, (s, r, score) in enumerate(zip(samples, responses, scores))
    ]
    return TaskResult(
        elapsed_seconds=round(elapsed, 2),
        metrics=Metrics(exact_match=metric_value, exact_match_stderr=stderr),
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
        Tuple of (cached: bool, hf_home: Path) where cached indicates if dataset
        is in cache and hf_home is the HuggingFace cache directory.
    """
    from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

    hub_path = (
        Path(HF_HUB_CACHE)
        / f"datasets--{dataset.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    cached = hub_path.is_dir()

    if cached:
        old = ds_config.HF_HUB_OFFLINE
        ds_config.HF_HUB_OFFLINE = True
    try:
        yield cached, Path(HF_HOME)
    finally:
        if cached:
            ds_config.HF_HUB_OFFLINE = old
