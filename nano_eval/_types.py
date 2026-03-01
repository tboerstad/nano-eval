"""Shared types and constants used by both __init__ and core."""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

# Defaults — single source of truth for CLI, evaluate(), and ApiConfig.
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_EXTRA_REQUEST_PARAMS = "temperature=0,max_tokens=256,seed=42"


class CancellationToken:
    """Cooperative cancellation token for graceful shutdown on Ctrl+C.

    First SIGINT sets the token, allowing in-flight requests to finish.
    Second SIGINT falls through to default behaviour (hard exit).
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event: asyncio.Event | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if self._event is not None:
            self._event.set()

    def _get_event(self) -> asyncio.Event:
        """Return an asyncio.Event that fires when cancel() is called."""
        if self._event is None:
            self._event = asyncio.Event()
            if self._cancelled:
                self._event.set()
        return self._event


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
    tokens_per_second: float


class EvalResult(TypedDict):
    cancelled: bool
    config: dict[str, Any]
    framework_version: str
    results: dict[str, TaskResult]
    total_seconds: float
