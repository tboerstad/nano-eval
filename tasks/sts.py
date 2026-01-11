"""
STS-B evaluation - semantic textual similarity benchmark.

Defines:
- samples(): load sentence pairs with similarity labels
- score(): placeholder (embedding scoring uses Spearman correlation in run_task)
- sts_b: Task instance for registration
"""

from __future__ import annotations

import logging

from core import Sample, Task, offline_if_cached

logger = logging.getLogger(__name__)

_STSB_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"


def samples(max_samples: int | None = None, seed: int | None = None) -> list[Sample]:
    """Load STS-B samples: ((sentence1, sentence2), normalized_similarity)."""
    import datasets
    from datasets import Dataset, DownloadMode

    datasets.utils.logging.set_verbosity_error()

    with offline_if_cached("sentence-transformers/stsb", _STSB_REVISION) as (
        cached,
        hf_home,
    ):
        logger.info(
            f"Cache {'hit' if cached else 'miss'} for embedding dataset (STS-B), "
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
                similarity = doc["score"] / 5.0
                result.append(
                    Sample(
                        prompt=(doc["sentence1"], doc["sentence2"]),
                        target=str(similarity),
                    )
                )
            if max_samples is not None:
                remaining = max_samples - len(result)
        return result


def score(response: float, target: str) -> float:
    """Placeholder score function (embedding tasks use Spearman correlation)."""
    return float(response) if isinstance(response, (int, float)) else 0.0


sts_b = Task(
    name="sts_b",
    task_type="embedding",
    samples=samples,
    score=score,
)
