"""
End-to-end tests for nano-eval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import hashlib
import json
from unittest.mock import patch

import httpx
import respx
from click.testing import CliRunner
from httpx import Response

from nano_eval import main
from nano_eval.tasks.chartqa import chartqa
from nano_eval.tasks.gsm8k import gsm8k_cot_llama

# GSM8K: 10 mock responses keyed by prompt hash (8 correct, 2 wrong = 80% accuracy)
# Hashes are for the last user message (the question) in multiturn fewshot format
# fmt: off
GSM8K_RESPONSES = {
    "53ef58": "The final answer is $3",     # ✓ target=3 (dollar sign)
    "74bdfd": "The final answer is 18.",    # ✓ target=18 (trailing period)
    "0f59d8": "Hmm, 8 times 8 is 64",      # ✓ target=64 (fallback extraction, no "The final answer is")
    "3da534": "The final answer is 20",     # ✓ target=20
    "3adba7": "The final answer is 999",    # ✗ target=45
    "ca5057": "I first thought The final answer is 999, but I made an error. The final answer is 540",  # ✓ target=540 (last-match)
    "61fb92": "The final answer is 999",    # ✗ target=160
    "00b394": "The final answer is $460",   # ✓ target=460 (dollar sign)
    "021f4b": "The final answer is 260",    # ✓ target=260
    "c58df6": "The final answer is 70,000", # ✓ target=70000 (comma-formatted)
}
# fmt: on

# ChartQA: 10 mock responses keyed by prompt hash (8 correct, 2 wrong = 80% accuracy)
# fmt: off
CHARTQA_RESPONSES = {
    "beb96e": "Final Answer: 6",      # ✓ target=6
    "5349c3": "**Final Answer:** 3",  # ✓ target=3 (markdown bold)
    "173532": "Final Answer: No",     # ✓ target=No
    "9e8414": "Final Answer: 14",     # ✓ target=14
    "15ac1f": "Based on analysis, the answer: 62.",  # ✓ target=62 (trailing punct, non-standard prefix)
    "2f2353": "Looking at the chart: FINAL ANSWER: 999. Wait, I need to recount... FINAL ANSWER: 23",  # ✓ target=23 (last-match)
    "5894e6": "FINAL ANSWER: wrong",  # ✗ target=Yes
    "509bd4": "Final Answer: 3%",     # ✓ target=0.03 (percent/decimal equivalence)
    "bad2cb": "Final Answer: 0.57",   # ✓ target=0.57
    "9addd0": "FINAL ANSWER: wrong",  # ✗ target=Inspired
}
# fmt: on

GSM8K_HASH = "7e9d77f0de73bfe63bd9858b220417f93da9f99a1a3e9b8e248c5f87eab9ec6d"
CHARTQA_HASH = "fc630d612ba41511d4d211d6849f8aec066be7f84dee4b2ba50fc2c5987b0af1"


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with real dataset, mocked API, auto-selected model."""
        real_samples = gsm8k_cot_llama.load_samples(10)

        def api_response(request):
            body = json.loads(request.content)
            # Verify default generation parameters are passed
            assert body["temperature"] == 0
            assert body["max_tokens"] == 256
            assert body["seed"] == 42
            # Extract last user message for multiturn fewshot format
            last_user_msg = [m for m in body["messages"] if m["role"] == "user"][-1]
            prompt = last_user_msg["content"]
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            if h not in GSM8K_RESPONSES:
                raise ValueError(f"Unknown prompt hash: {h}")
            content = GSM8K_RESPONSES[h]
            return Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        with respx.mock:
            respx.get("http://test.com/v1/chat/completions").mock(
                return_value=Response(200)
            )
            respx.get("http://test.com/v1/models").mock(
                return_value=Response(
                    200, json={"object": "list", "data": [{"id": "test"}]}
                )
            )
            respx.post("http://test.com/v1/chat/completions").mock(
                side_effect=api_response
            )

            with patch.object(
                type(gsm8k_cot_llama), "load_samples", return_value=real_samples
            ):
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    [
                        "--modality=text",
                        "--base-url=http://test.com/v1",
                        "--max-samples=10",
                        "--output-path",
                        str(tmp_path),
                        "--log-requests",
                    ],
                )
                assert result.exit_code == 0, result.output

        results = json.loads((tmp_path / "eval_results.json").read_text())
        assert results["results"]["text"]["metrics"]["accuracy"] == 0.8
        assert results["results"]["text"]["samples_hash"] == GSM8K_HASH

        requests = [
            json.loads(line)
            for line in (tmp_path / "request_log_text.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(requests) == 10
        assert requests[0]["request_id"] == 0
        assert requests[0]["target"] == "18"
        assert requests[0]["response"] == "The final answer is 18."
        assert requests[0]["score"] == 1.0
        assert requests[0]["stop_reason"] == "stop"
        assert requests[0]["output_tokens"] == 5
        assert "duration_seconds" in requests[0]
        assert isinstance(requests[0]["duration_seconds"], float)
        assert requests[3]["target"] == "540"
        assert requests[3]["score"] == 1.0

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with real dataset, mocked API."""
        real_samples = chartqa.load_samples(10)

        def api_response(request):
            body = json.loads(request.content)
            # Extract text from vision message content array
            content_list = body["messages"][0]["content"]
            prompt = next(c["text"] for c in content_list if c["type"] == "text")
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            if h not in CHARTQA_RESPONSES:
                raise ValueError(f"Unknown prompt hash: {h}")
            content = CHARTQA_RESPONSES[h]
            return Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )

        with respx.mock:
            respx.get("http://test.com/v1/chat/completions").mock(
                return_value=Response(200)
            )
            respx.post("http://test.com/v1/chat/completions").mock(
                side_effect=api_response
            )

            with patch.object(type(chartqa), "load_samples", return_value=real_samples):
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    [
                        "--modality=vision",
                        "--base-url=http://test.com/v1",
                        "--model=test",
                        "--max-samples=10",
                        "--output-path",
                        str(tmp_path),
                        "--log-requests",
                    ],
                )
                assert result.exit_code == 0, result.output

        results = json.loads((tmp_path / "eval_results.json").read_text())
        assert results["results"]["vision"]["metrics"]["accuracy"] == 0.8
        assert results["results"]["vision"]["samples_hash"] == CHARTQA_HASH

        requests = [
            json.loads(line)
            for line in (tmp_path / "request_log_vision.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(requests) == 10
        assert requests[0]["request_id"] == 0
        assert requests[0]["target"] == "14"
        assert requests[0]["response"] == "Final Answer: 14"
        assert requests[0]["score"] == 1.0
        assert requests[0]["stop_reason"] == "stop"
        assert requests[0]["output_tokens"] == 10
        assert "duration_seconds" in requests[0]
        assert isinstance(requests[0]["duration_seconds"], float)
        assert requests[4]["target"] == "23"
        assert requests[4]["score"] == 1.0

    def test_timeout_excluded_from_results(self, tmp_path):
        """Timed-out requests are excluded from results instead of crashing."""
        real_samples = gsm8k_cot_llama.load_samples(10)
        timeout_hashes = {"53ef58", "3adba7"}

        def api_response(request):
            body = json.loads(request.content)
            last_user_msg = [m for m in body["messages"] if m["role"] == "user"][-1]
            prompt = last_user_msg["content"]
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            if h in timeout_hashes:
                raise httpx.ReadTimeout("simulated timeout")
            content = GSM8K_RESPONSES[h]
            return Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        with respx.mock:
            respx.get("http://test.com/v1/chat/completions").mock(
                return_value=Response(200)
            )
            respx.get("http://test.com/v1/models").mock(
                return_value=Response(
                    200, json={"object": "list", "data": [{"id": "test"}]}
                )
            )
            respx.post("http://test.com/v1/chat/completions").mock(
                side_effect=api_response
            )

            with patch.object(
                type(gsm8k_cot_llama), "load_samples", return_value=real_samples
            ):
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    [
                        "--modality=text",
                        "--base-url=http://test.com/v1",
                        "--max-samples=10",
                        "--output-path",
                        str(tmp_path),
                        "--log-requests",
                    ],
                )
                assert result.exit_code == 0, result.output

        results = json.loads((tmp_path / "eval_results.json").read_text())
        task_result = results["results"]["text"]
        assert task_result["num_samples"] == 8

        requests = [
            json.loads(line)
            for line in (tmp_path / "request_log_text.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(requests) == 8


class TestCLI:
    """Tests for CLI behavior."""

    def test_help_startup_time(self):
        """CLI --help should complete quickly without importing heavy dependencies."""
        import subprocess
        import sys
        import time

        # Run multiple times to get a stable measurement
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-m", "nano_eval", "--help"],
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert result.returncode == 0, f"--help failed: {result.stderr}"

        median_time = sorted(times)[1]
        # Threshold: 0.8s allows headroom while catching regressions from heavy imports
        # (importing datasets/PIL/tqdm typically adds 1-2s)
        assert median_time < 0.8, (
            f"CLI --help took {median_time:.2f}s (median of {times}). "
            "This likely means heavy imports (datasets, PIL, etc.) are being loaded at "
            "module level instead of being deferred to evaluate()."
        )
