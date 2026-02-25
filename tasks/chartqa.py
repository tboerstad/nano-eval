"""ChartQA evaluation: relaxed matching with 5% numeric tolerance."""

from __future__ import annotations

import re
import string
from typing import Any

from core import Prompt, Sample, Task

_CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"


def _format_chartqa_prompt(query: str) -> str:
    return (
        f"{query}\n"
        "Analyze the image and question carefully, using step-by-step reasoning.\n"
        "First, describe any image provided in detail. "
        "Then, present your reasoning. "
        "And finally your final answer in this format:\n"
        "Final Answer: <answer>\n"
        "where <answer> follows the following instructions:\n"
        "- <answer> should be a single phrase or number.\n"
        "- <answer> should not paraphrase or reformat the text in the image.\n"
        "- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.\n"
        "- If the question is a Yes/No question, <answer> should be Yes/No.\n"
        "- If <answer> is a number, it should not contain any units.\n"
        "- If <answer> is a percentage, it should include a % sign.\n"
        "- If <answer> is an entity, it should include the full label from the graph.\n"
        "IMPORTANT: Remember, to end your answer with Final Answer: <answer>."
    )


def _get_final_answer(response: str) -> str:
    """Extract the final answer from a model response.

    Finds the last occurrence of 'answer:' (case-insensitive) and returns the
    first non-empty line following it. Uses rfind so reasoning models that
    self-correct are scored on their final answer, not an intermediate one.
    """
    # Normalize markdown emphasis around 'answer:' (e.g. **answer:** -> answer:)
    generation = re.sub(r"([aA]nswer)\**:\**", r"\1:", response)
    idx = generation.lower().rfind("answer:")
    if idx == -1:
        return ""
    start = idx + len("answer:")
    lines = generation[start:].split("\n")
    answer = next((line.strip() for line in lines if line.strip()), "")
    # Strip markdown formatting characters
    answer = re.sub(r"[*_\[\]()]", "", answer)
    return answer


def _preprocess_text(text: str) -> str:
    if not any(char.isdigit() for char in text):
        # Strip surrounding quotes for non-numeric text
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            return text[1:-1]
        return text
    else:
        # Strip trailing punctuation (but preserve %) and remove commas/$
        while (
            text
            and (text[-1] in string.punctuation or text[-1].isspace())
            and text[-1] != "%"
        ):
            text = text[:-1]
        return text.replace(",", "").replace("$", "")


def _to_float(text: str) -> tuple[float | None, bool]:
    text = text.strip()
    is_percent = text.endswith("%")
    try:
        value = float(text.rstrip("%"))
        return value, is_percent
    except ValueError:
        return None, False


def _score(response: str, target: str) -> float:
    pred = _get_final_answer(response)
    if not pred:
        return 0.0

    pred = _preprocess_text(pred)
    target = _preprocess_text(target)

    pred_float, pred_is_pct = _to_float(pred)
    target_float, target_is_pct = _to_float(target)

    if pred_float is not None and target_float is not None:

        def _rel_eq(a: float, b: float) -> bool:
            return abs(a - b) / max(abs(b), 1e-10) <= 0.05

        if _rel_eq(pred_float, target_float):
            return 1.0
        # Also try percent/decimal equivalence (e.g. "5%" vs "0.05")
        if pred_is_pct or target_is_pct:
            if _rel_eq(pred_float / 100 if pred_is_pct else pred_float, target_float):
                return 1.0
            if _rel_eq(
                pred_float, target_float / 100 if target_is_pct else target_float
            ):
                return 1.0
        return 0.0
    else:
        # Text comparison: strip trailing punctuation before comparing
        pred_text = pred
        while pred_text and pred_text[-1] in string.punctuation:
            pred_text = pred_text[:-1]
        return 1.0 if pred_text.lower() == target.lower() else 0.0


def _extract(doc: dict[str, Any]) -> Sample:
    label = doc["label"]
    target = label[0] if isinstance(label, list) else str(label)
    return Sample(
        prompt=Prompt(
            text=_format_chartqa_prompt(doc["query"]),
            images=[doc["image"]],
        ),
        target=target,
    )


chartqa = Task(
    name="chartqa",
    dataset="HuggingFaceM4/ChartQA",
    revision=_CHARTQA_REVISION,
    splits=["test", "val", "train"],
    extract=_extract,
    score=_score,
)
