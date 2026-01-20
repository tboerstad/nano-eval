"""
ChartQA evaluation - multimodal chart understanding.

Defines:
- samples(): generator yielding ((prompt, images), target) pairs
- score(): relaxed matching with 5% numeric tolerance
- chartqa: Task instance for registration
"""

from __future__ import annotations

import logging
import re

from core import Sample, Task, VisionPrompt, offline_if_cached

logger = logging.getLogger("nano_eval.tasks.chartqa")

_CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"

# Extracts answer after "FINAL ANSWER:" up to newline or end (prompt instructs model to use this)
# Non-greedy (.+?) stops at first newline to avoid capturing extra text
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Strip currency/percent symbols for numeric comparison: "$1,234%" -> "1234"
_NUMERIC_CLEAN_RE = re.compile(r"[$,%]")


def _format_chartqa_prompt(query: str) -> str:
    """Format ChartQA prompt."""
    return (
        f"<image>You are provided a chart image and will be asked a question. "
        f"You have to think through your answer and provide a step-by-step solution. "
        f'Once you have the solution, write the final answer in at most a few words at the end with the phrase "FINAL ANSWER:". '
        f"The question is: {query}\n"
        f"Let's think step by step."
    )


def score(response: str, target: str) -> float:
    """ChartQA relaxed match: exact match or 5% numeric tolerance."""
    if match := _FINAL_ANSWER_RE.search(response):
        pred = match.group(1).strip()
    else:
        pred = response.strip()

    if pred.lower() == target.lower():
        return 1.0

    try:
        pred_n = float(_NUMERIC_CLEAN_RE.sub("", pred))
        target_n = float(_NUMERIC_CLEAN_RE.sub("", target))
        if target_n == 0:
            return 1.0 if pred_n == 0 else 0.0
        if abs(pred_n - target_n) / abs(target_n) <= 0.05:
            return 1.0
    except ValueError:
        pass

    return 0.0


def samples(max_samples: int | None = None, seed: int | None = None) -> list[Sample]:
    """Load ChartQA samples: ((prompt, [image]), target)."""
    import datasets
    from datasets import Dataset, DownloadMode

    # TODO Upstream fix. HF datasets logging is too noisy
    datasets.utils.logging.set_verbosity_error()

    with offline_if_cached("HuggingFaceM4/ChartQA", _CHARTQA_REVISION) as (
        cached,
        hf_home,
    ):
        logger.info(
            f"Cache {'hit' if cached else 'miss'} for vision dataset (ChartQA), "
            f"HF_HOME={hf_home}"
        )
        result: list[Sample] = []
        remaining = max_samples
        for split in ["test", "val", "train"]:
            if remaining is not None and remaining <= 0:
                break
            ds = datasets.load_dataset(
                "HuggingFaceM4/ChartQA",
                split=split,
                revision=_CHARTQA_REVISION,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
            assert isinstance(ds, Dataset)
            if seed is not None:
                ds = ds.shuffle(seed=seed)
            if remaining is not None:
                ds = ds.select(range(min(remaining, len(ds))))
            for doc in ds:
                label = doc["label"]
                target = label[0] if isinstance(label, list) else str(label)
                result.append(
                    Sample(
                        prompt=VisionPrompt(
                            text=_format_chartqa_prompt(doc["query"]),
                            images=[doc["image"]],
                        ),
                        target=target,
                    )
                )
            if max_samples is not None:
                remaining = max_samples - len(result)
        return result


chartqa = Task(
    name="chartqa",
    task_type="vision",
    samples=samples,
    score=score,
)
