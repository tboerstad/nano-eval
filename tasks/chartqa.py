"""ChartQA evaluation task for multimodal chart understanding."""

from __future__ import annotations

import re
from typing import Any

from core import Sample, Task, load_samples

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


def _relaxed_match(response: str, target: str) -> float:
    """ChartQA metric: exact match or 5% numeric tolerance."""
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


def _transform(doc: Any) -> Sample:
    label = doc["label"]
    target = label[0] if isinstance(label, list) else str(label)
    return Sample(
        prompt=(_format_chartqa_prompt(doc["query"]), [doc["image"]]),
        target=target,
    )


def samples(max_samples: int | None = None, seed: int | None = None) -> list[Sample]:
    return load_samples(
        dataset="HuggingFaceM4/ChartQA",
        revision=_CHARTQA_REVISION,
        splits=["test", "val", "train"],
        transform=_transform,
        max_samples=max_samples,
        seed=seed,
    )


def score(response: str, target: str) -> float:
    """Score ChartQA response with relaxed matching (5% numeric tolerance)."""
    return _relaxed_match(response, target)


chartqa = Task(
    name="chartqa",
    task_type="vision",
    samples=samples,
    score=score,
)
