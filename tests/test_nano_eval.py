"""
End-to-end tests for nano-eval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import hashlib
import json
import sys
from unittest.mock import patch

import respx
from httpx import Response

from core import Task
from nano_eval import main
from tasks.chartqa import samples as load_chartqa_samples, score as chartqa_score
from tasks.gsm8k import samples as load_gsm8k_samples, score as gsm8k_score

# GSM8K: 10 mock responses keyed by prompt hash (7 correct, 3 wrong = 70% accuracy)
# Hashes are for the last user message (the question) in multiturn fewshot format
# fmt: off
GSM8K_RESPONSES = {
    "b4ae7b": "The final answer is 3",      # ✓ target=3
    "2d1210": "The final answer is 18",     # ✓ target=18
    "dbc356": "The final answer is 64",     # ✓ target=64
    "a1f1f0": "The final answer is 20",     # ✓ target=20
    "050045": "The final answer is 999",    # ✗ target=45
    "464d52": "The final answer is 999",    # ✗ target=540
    "64816f": "The final answer is 999",    # ✗ target=160
    "967e59": "The final answer is 460",    # ✓ target=460
    "8c5053": "The final answer is 260",    # ✓ target=260
    "d85409": "The final answer is 70000",  # ✓ target=70000
}
# fmt: on

# ChartQA: 10 mock responses keyed by prompt hash (7 correct, 3 wrong = 70% accuracy)
# fmt: off
CHARTQA_RESPONSES = {
    "0ad231": "FINAL ANSWER: 6",      # ✓ target=6
    "e4b999": "FINAL ANSWER: 3",      # ✓ target=3
    "46d9da": "FINAL ANSWER: No",     # ✓ target=No
    "4b62c8": "FINAL ANSWER: 14",     # ✓ target=14
    "c362e0": "FINAL ANSWER: 62",     # ✓ target=62
    "cc97c4": "FINAL ANSWER: 999",    # ✗ target=23
    "85801e": "FINAL ANSWER: wrong",  # ✗ target=Yes
    "ae07dd": "FINAL ANSWER: 0.03",   # ✓ target=0.03
    "c26baf": "FINAL ANSWER: 0.57",   # ✓ target=0.57
    "72eade": "FINAL ANSWER: wrong",  # ✗ target=Inspired
}
# fmt: on

GSM8K_HASH = "1330276a7b9c8140e39e0d966882feb6898dba391dd376c639c2b9d5cbe0464e"
CHARTQA_HASH = "c6043b6621aa01df9276f665b2a8a1be6ac28c86caf5822ae3b4333a4b793caf"


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with real dataset, mocked API, auto-selected model."""
        real_samples = load_gsm8k_samples(10)

        def api_response(request):
            body = json.loads(request.content)
            # Extract last user message for multiturn fewshot format
            last_user_msg = [m for m in body["messages"] if m["role"] == "user"][-1]
            prompt = last_user_msg["content"]
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            if h not in GSM8K_RESPONSES:
                raise ValueError(f"Unknown prompt hash: {h}")
            content = GSM8K_RESPONSES[h]
            return Response(200, json={"choices": [{"message": {"content": content}}]})

        task = Task(
            name="gsm8k_cot_llama",
            samples=lambda n, seed: real_samples,
            score=gsm8k_score,
        )

        with respx.mock:
            respx.get("http://test.com/v1/models").mock(
                return_value=Response(
                    200, json={"object": "list", "data": [{"id": "test"}]}
                )
            )
            respx.post("http://test.com/v1/chat/completions").mock(
                side_effect=api_response
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "nano-eval",
                        "--tasks",
                        "gsm8k_cot_llama",
                        "--base_url",
                        "http://test.com/v1",
                        "--max_samples",
                        "10",
                        "--output_path",
                        str(tmp_path),
                        "--log_samples",
                    ],
                ),
                patch.dict("tasks.TASKS", {"gsm8k_cot_llama": task}),
            ):
                main()

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["gsm8k_cot_llama"]["metrics"]["exact_match"] == 0.7
        assert results["results"]["gsm8k_cot_llama"]["task_hash"] == GSM8K_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_gsm8k_cot_llama.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["sample_id"] == 0
        assert samples[0]["target"] == "18"
        assert samples[0]["response"] == "The final answer is 18"
        assert samples[0]["exact_match"] == 1.0
        assert samples[3]["target"] == "540"
        assert samples[3]["exact_match"] == 0.0

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with real dataset, mocked API."""
        real_samples = load_chartqa_samples(10)

        def api_response(request):
            body = json.loads(request.content)
            # Extract text from vision message content array
            content_list = body["messages"][0]["content"]
            prompt = next(c["text"] for c in content_list if c["type"] == "text")
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            if h not in CHARTQA_RESPONSES:
                raise ValueError(f"Unknown prompt hash: {h}")
            content = CHARTQA_RESPONSES[h]
            return Response(200, json={"choices": [{"message": {"content": content}}]})

        task = Task(
            name="chartqa", samples=lambda n, seed: real_samples, score=chartqa_score
        )

        with respx.mock:
            respx.post("http://test.com/v1/chat/completions").mock(
                side_effect=api_response
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "nano-eval",
                        "--tasks",
                        "chartqa",
                        "--base_url",
                        "http://test.com/v1",
                        "--model",
                        "test",
                        "--max_samples",
                        "10",
                        "--output_path",
                        str(tmp_path),
                        "--log_samples",
                    ],
                ),
                patch.dict("tasks.TASKS", {"chartqa": task}),
            ):
                main()

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["chartqa"]["metrics"]["exact_match"] == 0.7
        assert results["results"]["chartqa"]["task_hash"] == CHARTQA_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_chartqa.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["sample_id"] == 0
        assert samples[0]["target"] == "14"
        assert samples[0]["response"] == "FINAL ANSWER: 14"
        assert samples[0]["exact_match"] == 1.0
        assert samples[4]["target"] == "23"
        assert samples[4]["exact_match"] == 0.0
