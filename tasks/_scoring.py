"""Shared extraction and normalization helpers for task scoring."""

from __future__ import annotations

import re

_NUMERIC_JUNK_RE = re.compile(r"[$,%]")


def extract_final_answer(response: str, pattern: re.Pattern[str]) -> str | None:
    """Return group(1) from the first *pattern* match in *response*, or ``None``."""
    if match := pattern.search(response):
        return match.group(1).strip()
    return None


def clean_number(text: str) -> str:
    """Strip currency / percentage formatting (``$``, ``,``, ``%``)."""
    return _NUMERIC_JUNK_RE.sub("", text)
