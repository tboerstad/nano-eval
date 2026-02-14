"""ChartQA evaluation: relaxed matching with 5% numeric tolerance."""

import re

from core import Prompt, Sample, Task

_CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_NUMERIC_CLEAN_RE = re.compile(r"[$,%]")


def _format_chartqa_prompt(query):
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


def _score(response, target):
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


def _extract(doc):
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
