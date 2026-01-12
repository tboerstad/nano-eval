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


class AccuracyMetrics(TypedDict):
    """Metrics for text/vision tasks using exact match scoring."""

    exact_match: float
    exact_match_stderr: float


class SpearmanMetrics(TypedDict):
    """Metrics for embedding tasks using Spearman correlation."""

    spearman_correlation: float
    spearman_correlation_stderr: float


Metrics = AccuracyMetrics | SpearmanMetrics


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


@dataclass(frozen=True)
class TextPrompt:
    """Text-only prompt (simple string or pre-formatted messages)."""

    text: str | list[dict[str, str]]


@dataclass(frozen=True)
class VisionPrompt:
    """Multimodal prompt with text and images."""

    text: str
    images: list[Any]


@dataclass(frozen=True)
class EmbeddingPrompt:
    """Pair of texts for embedding similarity evaluation."""

    sentence1: str
    sentence2: str


Input = TextPrompt | VisionPrompt | EmbeddingPrompt


@dataclass
class Sample:
    """A single evaluation sample: prompt + expected target."""

    prompt: Input
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
    prompts: list[Input],
    config: APIConfig,
    progress_desc: str = "Running evals",
) -> list[str]:
    """
    Run batch of chat completions.

    Args:
        prompts: List of prompts (TextPrompt or VisionPrompt)
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
            if isinstance(prompt, VisionPrompt):
                messages = _build_vision_message(prompt.text, prompt.images)
            elif isinstance(prompt, TextPrompt):
                if isinstance(prompt.text, list):
                    messages = prompt.text
                else:
                    messages = [{"role": "user", "content": prompt.text}]
            else:
                raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")

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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)

    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n

    num = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    den_x = math.sqrt(sum((rx - mean_x) ** 2 for rx in rank_x))
    den_y = math.sqrt(sum((ry - mean_y) ** 2 for ry in rank_y))

    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _spearman_stderr(r: float, n: int) -> float:
    """Compute standard error for Spearman correlation using Fisher z-transform."""
    if n < 4:
        return 0.0
    z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else 0.0
    se_z = 1.0 / math.sqrt(n - 3)
    r_lower = math.tanh(z - 1.96 * se_z)
    r_upper = math.tanh(z + 1.96 * se_z)
    return (r_upper - r_lower) / (2 * 1.96)


async def _embed_batch(
    client: httpx.AsyncClient,
    url: str,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Embed a batch of texts. Returns list of embedding vectors."""
    payload = {"model": model, "input": texts}
    resp = await client.post(url, json=payload)
    if not resp.is_success:
        raise RuntimeError(f"Embedding request failed: {resp.text}")
    data = resp.json()["data"]
    sorted_data = sorted(data, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


async def embed(
    prompts: list[EmbeddingPrompt],
    config: APIConfig,
    progress_desc: str = "Running embedding eval",
) -> list[float]:
    """
    Embed sentence pairs and compute cosine similarities.

    Args:
        prompts: List of EmbeddingPrompt (sentence pairs)
        config: API configuration

    Returns:
        List of cosine similarities (one per pair)
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = config.url.replace("/chat/completions", "/embeddings")

    all_sentences = [s for p in prompts for s in (p.sentence1, p.sentence2)]

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.max_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
        batch_size = 32
        all_embeddings: list[list[float]] = []
        tasks: list[asyncio.Task[list[list[float]]]] = []

        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i : i + batch_size]
            tasks.append(
                asyncio.create_task(_embed_batch(client, url, batch, config.model))
            )

        try:
            batches = await tqdm_asyncio.gather(*tasks, desc=progress_desc, leave=False)
            for batch_result in batches:
                all_embeddings.extend(batch_result)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    return [
        _cosine_similarity(all_embeddings[i], all_embeddings[i + 1])
        for i in range(0, len(all_embeddings), 2)
    ]


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


def _prompt_to_str(prompt: Input) -> str:
    """Extract text from prompt (handles TextPrompt, VisionPrompt, EmbeddingPrompt)."""
    if isinstance(prompt, VisionPrompt):
        return prompt.text
    if isinstance(prompt, EmbeddingPrompt):
        return f"{prompt.sentence1} ||| {prompt.sentence2}"
    if isinstance(prompt.text, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in prompt.text)
    return prompt.text


def compute_samples_hash(samples: list[Sample]) -> str:
    """Compute SHA256 hash for all samples in a task (includes image data)."""
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(_prompt_to_str(s.prompt).encode())
        if isinstance(s.prompt, VisionPrompt):
            for img in s.prompt.images:
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

    if task.task_type == "embedding":
        embedding_prompts = [p for p in prompts if isinstance(p, EmbeddingPrompt)]
        assert len(embedding_prompts) == len(prompts), "Expected all EmbeddingPrompt"
        similarities = await embed(embedding_prompts, config, "Running embedding eval")
        responses = [f"{sim:.6f}" for sim in similarities]
        elapsed = time.perf_counter() - t0

        targets = [float(s.target) for s in samples]
        correlation = _spearman_correlation(similarities, targets)
        stderr = _spearman_stderr(correlation, len(samples))
        scores = [task.score(r, s.target) for r, s in zip(responses, samples)]

        logger.debug(
            f"{task.name}: spearman={correlation:.4f}±{stderr:.4f} ({elapsed:.2f}s)"
        )

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
            metrics=SpearmanMetrics(
                spearman_correlation=correlation, spearman_correlation_stderr=stderr
            ),
            num_samples=len(samples),
            samples=logged_samples,
            samples_hash=samples_hash,
            task=task.name,
            task_type=task.task_type,
        )

    desc = "Running vision eval" if task.task_type == "vision" else "Running text eval"
    responses = await complete(prompts, config, desc)
    elapsed = time.perf_counter() - t0

    scores = [task.score(r, s.target) for r, s in zip(responses, samples)]
    n = len(samples)
    accuracy = sum(scores) / n if n else 0.0
    stderr = math.sqrt(accuracy * (1 - accuracy) / (n - 1)) if n > 1 else 0.0

    logger.debug(f"{task.name}: accuracy={accuracy:.4f}±{stderr:.4f} ({elapsed:.2f}s)")

    logged_samples = [
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
        metrics=AccuracyMetrics(exact_match=accuracy, exact_match_stderr=stderr),
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
