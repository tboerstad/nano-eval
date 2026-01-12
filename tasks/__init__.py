"""
Task registry for nano-eval.

Users can add custom tasks to TASKS dict.
"""

from core import Task

TASKS: dict[str, Task] = {}

__all__ = ["TASKS", "Task"]
