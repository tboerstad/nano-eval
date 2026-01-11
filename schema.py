"""Type definitions for nano-eval results.

All result types are defined here to allow lightweight imports without
triggering heavy dependencies (datasets, PIL, tqdm) from core.py.
"""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict

__all__ = ["ConfigInfo", "EvalResult", "LoggedSample", "Metrics", "TaskResult"]


class Metrics(TypedDict):
    """Evaluation metrics for a task."""

    exact_match: float
    exact_match_stderr: float


class LoggedSample(TypedDict):
    """Per-sample evaluation data for JSONL export."""

    sample_id: int
    target: str
    prompt: str
    response: str
    exact_match: float


class TaskResult(TypedDict):
    """Result from evaluating a single task."""

    elapsed_seconds: float
    metrics: Metrics
    num_samples: int
    samples: NotRequired[list[LoggedSample]]
    samples_hash: str
    task: str
    task_type: str


class ConfigInfo(TypedDict):
    """Configuration metadata stored in results."""

    model: str
    max_samples: int | None


class EvalResult(TypedDict):
    """Top-level evaluation result returned by evaluate()."""

    config: ConfigInfo
    framework_version: str
    results: dict[str, TaskResult]
    total_seconds: float
