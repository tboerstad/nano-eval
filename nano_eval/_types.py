"""Shared types and constants used by both __init__ and core."""

from __future__ import annotations

from typing import Any, TypedDict

# Defaults — single source of truth for CLI, evaluate(), and ApiConfig.
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_EXTRA_REQUEST_PARAMS = "temperature=0,max_tokens=256,seed=42"


class Metrics(TypedDict):
    accuracy: float
    accuracy_stderr: float


class TaskResult(TypedDict):
    elapsed_seconds: float
    metrics: Metrics
    num_samples: int
    samples_hash: str
    task: str
    modality: str
    total_input_tokens: int
    total_output_tokens: int
    mean_request_throughput: float


class EvalResult(TypedDict):
    config: dict[str, Any]
    framework_version: str
    results: dict[str, TaskResult]
    total_seconds: float
