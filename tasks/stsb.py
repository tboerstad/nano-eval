"""
STS Benchmark evaluation - semantic textual similarity.

Defines:
- samples(): generator yielding (sentence_pair, target_similarity) pairs
- score(): absolute error between predicted and target similarity
- stsb: Task instance for registration
"""

from __future__ import annotations

import logging

from core import EmbeddingPrompt, Sample, Task, offline_if_cached

logger = logging.getLogger(__name__)

_STSB_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"


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
)
