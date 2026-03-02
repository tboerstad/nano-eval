"""ChartQA evaluation: relaxed matching with 5% numeric tolerance."""

from __future__ import annotations

import re
import string
from typing import Any

from nano_eval.core import Prompt, Sample, Task

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
        "IMPORTANT: Remember, to end your answer with Final Answer: <answer>.\n"
    )


# ---------------------------------------------------------------------------
# Scoring vendored from lm-evaluation-harness (EleutherAI/lm-evaluation-harness)
# Source: lm_eval/tasks/chartqa/utils.py
# ---------------------------------------------------------------------------


def _normalize_string(s: str) -> str:
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(text: str) -> str:
    while (
        text
        and (text[-1] in string.punctuation or text[-1].isspace())
        and text[-1] != "%"
    ):
        text = text[:-1]
    return text


class _RelaxedCorrectness:
    """Relaxed correctness with 5% numeric tolerance.

    See https://arxiv.org/pdf/2203.10244.pdf, section 5.1.
    """

    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str) -> tuple[float | None, bool]:
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                return float(text.rstrip("%")), is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(char.isdigit() for char in text):
                return _normalize_string(text)
            return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def _relative_change(a: float, b: float) -> float:
            return abs(a - b) / max(abs(b), 1e-10)

        def _compare_numeric(a: float, b: float, max_rel: float) -> float:
            return 1.0 if _relative_change(a, b) <= max_rel else 0.0

        def _compare_text(pred: str, tgt: str) -> float:
            while pred and pred[-1] in string.punctuation:
                pred = pred[:-1]
            return 1.0 if pred.lower() == tgt.lower() else 0.0

        def _to_decimal(value: float, is_percent: bool) -> float:
            return value / 100 if is_percent else value

        def _compare_numeric_with_percent(
            pred: float,
            pred_pct: bool,
            tgt: float,
            tgt_pct: bool,
            max_rel: float,
        ) -> float:
            value = _compare_numeric(pred, tgt, max_rel)
            if value != 1.0 and (pred_pct or tgt_pct):
                value = max(
                    value,
                    _compare_numeric(_to_decimal(pred, pred_pct), tgt, max_rel),
                    _compare_numeric(pred, _to_decimal(tgt, tgt_pct), max_rel),
                )
            return value

        prediction = _preprocess_text(prediction)
        pred_float, pred_pct = _to_float(prediction)

        values: list[float] = []
        for target in targets:
            target = _preprocess_text(target)
            tgt_float, tgt_pct = _to_float(target)

            if pred_float is not None and tgt_float is not None:
                value = _compare_numeric_with_percent(
                    pred_float, pred_pct, tgt_float, tgt_pct, max_relative_change
                )
            elif _is_letter(target) and len(prediction) > 0:
                value = 1.0 if prediction[0].lower() == target.lower() else 0.0
            else:
                value = _compare_text(prediction, target)

            values.append(value)

        return max(values)

    def score(self, model_answer: str, reference: str | list[str]) -> float:
        ref_list = reference if isinstance(reference, list) else [reference]
        return self._relaxed_correctness(model_answer, ref_list)


class _ExplicitPromptRelaxedCorrectness(_RelaxedCorrectness):
    @staticmethod
    def _get_final_answer(generation: str) -> str:
        generation = re.sub(r"([aA]nswer)\**:\**", r"\1:", generation)
        idx = generation.lower().rfind("answer:")
        if idx == -1:
            return ""
        start = idx + len("answer:")
        lines = generation[start:].split("\n")
        answer = next((line.strip() for line in lines if line.strip()), "")
        return re.sub(r"[*_\[\]\(\)]", "", answer)

    def score(self, model_answer: str, reference: str | list[str]) -> float:
        parsed = self._get_final_answer(model_answer)
        if not parsed:
            return 0.0
        return super().score(parsed, reference)


_SCORER = _ExplicitPromptRelaxedCorrectness()


def _score(response: str, target: str) -> float:
    return _SCORER.score(response, target)


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
