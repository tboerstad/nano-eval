"""
GSM8K evaluation - grade school math with chain-of-thought.

Defines:
- samples(): generator yielding (prompt, target) pairs
- score(): normalized string matching
- gsm8k_cot_llama: Task instance for registration
"""

from __future__ import annotations

import logging
import re

from core import Sample, Task, TextPrompt, _normalize, offline_if_cached

logger = logging.getLogger("nano_eval.tasks.gsm8k")

_GSM8K_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"

GSM8K_FEWSHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    ),
]

_GSM8K_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
)

# Matches numbers in two forms:
# - "-?[$0-9.,]{2,}": formatted numbers like "$1,234.56" (2+ chars to avoid lone punctuation)
# - "-?[0-9]+": simple integers like "5" (catches single digits the first pattern misses)
_NUM_RE = re.compile(r"-?[$0-9.,]{2,}|-?[0-9]+")


def _format_gsm8k_prompt(question: str) -> list[dict[str, str]]:
    """Format GSM8K prompt with few-shot examples as multiturn messages."""
    messages: list[dict[str, str]] = []
    for q, a in GSM8K_FEWSHOT:
        messages.append({"role": "user", "content": _GSM8K_TEMPLATE.format(question=q)})
        messages.append({"role": "assistant", "content": a})
    messages.append(
        {"role": "user", "content": _GSM8K_TEMPLATE.format(question=question)}
    )
    return messages


def _parse_target(answer: str) -> str:
    """Parse target answer from GSM8K format, handling missing #### delimiter."""
    parts = answer.split("####")
    if len(parts) < 2:
        return answer.strip()
    return parts[-1].strip()


# Extracts number from "The final answer is 42" format (prompt instructs model to use this)
_FINAL_ANSWER_RE = re.compile(rf"The final answer is ({_NUM_RE.pattern})")


def _extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response."""
    if match := _FINAL_ANSWER_RE.search(response):
        return match.group(1)
    matches = _NUM_RE.findall(response)
    if matches:
        return matches[-1]
    return response


def samples(max_samples: int | None = None, seed: int | None = None) -> list[Sample]:
    """Load GSM8K samples: (formatted_prompt, target_answer)."""
    import datasets
    from datasets import Dataset, DownloadMode

    # TODO Upstream fix. HF datasets logging is too noisy
    datasets.utils.logging.set_verbosity_error()

    with offline_if_cached("gsm8k", _GSM8K_REVISION) as (cached, hf_home):
        logger.info(
            f"Cache {'hit' if cached else 'miss'} for text dataset (gsm8k_cot_llama), "
            f"HF_HOME={hf_home}"
        )
        result: list[Sample] = []
        remaining = max_samples
        for split in ["test", "train"]:
            if remaining is not None and remaining <= 0:
                break
            ds = datasets.load_dataset(
                "gsm8k",
                "main",
                split=split,
                revision=_GSM8K_REVISION,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
            assert isinstance(ds, Dataset)
            if seed is not None:
                ds = ds.shuffle(seed=seed)
            if remaining is not None:
                ds = ds.select(range(min(remaining, len(ds))))
            for doc in ds:
                result.append(
                    Sample(
                        prompt=TextPrompt(text=_format_gsm8k_prompt(doc["question"])),
                        target=_parse_target(doc["answer"]),
                    )
                )
            if max_samples is not None:
                remaining = max_samples - len(result)
        return result


def score(response: str, target: str) -> float:
    """Score GSM8K response: 1.0 if normalized answer matches, else 0.0."""
    extracted = _extract_gsm8k_answer(response)
    return 1.0 if _normalize(extracted) == _normalize(target) else 0.0


gsm8k_cot_llama = Task(
    name="gsm8k_cot_llama",
    task_type="text",
    samples=samples,
    score=score,
)
