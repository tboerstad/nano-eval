"""
End-to-end tests for nano-eval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import hashlib
import json
from unittest.mock import patch

import respx
from click.testing import CliRunner
from httpx import Response

from nano_eval import main
from tasks.chartqa import chartqa
from tasks.gsm8k import gsm8k_cot_llama

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
    "eed350": "FINAL ANSWER: 6",      # ✓ target=6
    "c6ad9c": "FINAL ANSWER: 3",      # ✓ target=3
    "f9f65c": "FINAL ANSWER: No",     # ✓ target=No
    "fa51ed": "FINAL ANSWER: 14",     # ✓ target=14
    "8e6330": "FINAL ANSWER: 62",     # ✓ target=62
    "b2bd79": "FINAL ANSWER: 999",    # ✗ target=23
    "372e0a": "FINAL ANSWER: wrong",  # ✗ target=Yes
    "833fc2": "FINAL ANSWER: 0.03",   # ✓ target=0.03
    "59f1fa": "FINAL ANSWER: 0.57",   # ✓ target=0.57
    "a14263": "FINAL ANSWER: wrong",  # ✗ target=Inspired
}
# fmt: on

GSM8K_HASH = "1330276a7b9c8140e39e0d966882feb6898dba391dd376c639c2b9d5cbe0464e"
CHARTQA_HASH = "8df185292f416992aeb99cd981f041421128de58c736ba17e7a1fadc2acf3f7e"


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with real dataset, mocked API, auto-selected model."""
        real_samples = gsm8k_cot_llama.load_samples(10)

        def api_response(request):
            body = json.loads(request.content)
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
        assert results["results"]["text"]["metrics"]["accuracy"] == 0.7
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
        assert requests[0]["response"] == "The final answer is 18"
        assert requests[0]["score"] == 1.0
        assert requests[0]["stop_reason"] == "stop"
        assert requests[0]["output_tokens"] == 5
        assert "duration_seconds" in requests[0]
        assert isinstance(requests[0]["duration_seconds"], float)
        assert requests[3]["target"] == "540"
        assert requests[3]["score"] == 0.0

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with real dataset, mocked API."""
        real_samples = chartqa.load_samples(10)

        def api_response(request):
            body = json.loads(request.content)
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
        assert results["results"]["vision"]["metrics"]["accuracy"] == 0.7
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
        assert requests[0]["response"] == "FINAL ANSWER: 14"
        assert requests[0]["score"] == 1.0
        assert requests[0]["stop_reason"] == "stop"
        assert requests[0]["output_tokens"] == 10
        assert "duration_seconds" in requests[0]
        assert isinstance(requests[0]["duration_seconds"], float)
        assert requests[4]["target"] == "23"
        assert requests[4]["score"] == 0.0


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
        # Threshold: 0.5s allows headroom while catching regressions from heavy imports
        # (importing datasets/PIL/tqdm typically adds 1-2s)
        assert median_time < 0.5, (
            f"CLI --help took {median_time:.2f}s (median of {times}). "
            "This likely means heavy imports (datasets, PIL, etc.) are being loaded at "
            "module level instead of being deferred to evaluate()."
        )
