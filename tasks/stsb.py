"""
STS Benchmark evaluation - semantic textual similarity.

Defines:
- samples(): generator yielding (sentence_pair, target_similarity) pairs
- score(): absolute error between predicted and target similarity
- embed(): async inference for embedding pairs
- compute_metrics(): Spearman correlation metrics
- stsb: Task instance for registration
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any

import httpx
from tqdm.asyncio import tqdm_asyncio

from core import (
    EmbeddingPrompt,
    Input,
    Sample,
    SpearmanMetrics,
    Task,
    offline_if_cached,
)

if TYPE_CHECKING:
    from core import APIConfig, Metrics

logger = logging.getLogger(__name__)

_STSB_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"


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


async def _embed_pair(
    client: httpx.AsyncClient,
    url: str,
    prompt: EmbeddingPrompt,
    model: str,
) -> str:
    """Embed a sentence pair and return cosine similarity as string."""
    payload = {"model": model, "input": [prompt.sentence1, prompt.sentence2]}
    resp = await client.post(url, json=payload)
    if not resp.is_success:
        raise RuntimeError(f"Embedding request failed: {resp.text}")
    data = resp.json()["data"]
    sorted_data = sorted(data, key=lambda x: x["index"])
    emb1, emb2 = sorted_data[0]["embedding"], sorted_data[1]["embedding"]
    return f"{_cosine_similarity(emb1, emb2):.6f}"


async def embed(prompts: list[Input], config: APIConfig) -> list[str]:
    """
    Embed sentence pairs and compute cosine similarities.

    Args:
        prompts: List of EmbeddingPrompt (sentence pairs)
        config: API configuration

    Returns:
        List of cosine similarities as strings (one per pair)
    """
    headers: dict[str, Any] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = config.url.replace("/chat/completions", "/embeddings")

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.max_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
        tasks = []
        for prompt in prompts:
            if not isinstance(prompt, EmbeddingPrompt):
                raise TypeError(
                    f"Expected EmbeddingPrompt, got {type(prompt).__name__}"
                )
            tasks.append(
                asyncio.create_task(_embed_pair(client, url, prompt, config.model))
            )

        try:
            return list(
                await tqdm_asyncio.gather(
                    *tasks, desc="Running embedding eval", leave=False
                )
            )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def compute_metrics(responses: list[str], samples: list[Sample]) -> Metrics:
    """Compute Spearman correlation metrics for embedding task."""
    predicted = [float(r) for r in responses]
    targets = [float(s.target) for s in samples]
    n = len(samples)
    corr = _spearman_correlation(predicted, targets)
    return SpearmanMetrics(
        spearman_correlation=corr,
        spearman_correlation_stderr=_spearman_stderr(corr, n),
    )


def samples(max_samples: int | None = None, seed: int | None = None) -> list[Sample]:
    """Load STS Benchmark samples: (sentence_pair, normalized_similarity)."""
    import datasets
    from datasets import Dataset, DownloadMode

    datasets.utils.logging.set_verbosity_error()

    with offline_if_cached("sentence-transformers/stsb", _STSB_REVISION) as (
        cached,
        hf_home,
    ):
        logger.info(
            f"Cache {'hit' if cached else 'miss'} for embedding dataset (stsb), "
            f"HF_HOME={hf_home}"
        )
        result: list[Sample] = []
        remaining = max_samples
        for split in ["test", "validation", "train"]:
            if remaining is not None and remaining <= 0:
                break
            ds = datasets.load_dataset(
                "sentence-transformers/stsb",
                split=split,
                revision=_STSB_REVISION,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
            assert isinstance(ds, Dataset)
            if seed is not None:
                ds = ds.shuffle(seed=seed)
            if remaining is not None:
                ds = ds.select(range(min(remaining, len(ds))))
            for doc in ds:
                normalized_score = doc["score"] / 5.0
                result.append(
                    Sample(
                        prompt=EmbeddingPrompt(
                            sentence1=doc["sentence1"],
                            sentence2=doc["sentence2"],
                        ),
                        target=f"{normalized_score:.6f}",
                    )
                )
            if max_samples is not None:
                remaining = max_samples - len(result)
        return result


def score(response: str, target: str) -> float:
    """Score as 1 - absolute error (higher is better, max 1.0)."""
    try:
        pred = float(response)
        tgt = float(target)
        return 1.0 - abs(pred - tgt)
    except ValueError:
        return 0.0


stsb = Task(
    name="stsb",
    task_type="embedding",
    samples=samples,
    score=score,
    infer=embed,
    compute_metrics=compute_metrics,
)
