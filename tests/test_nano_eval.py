"""
End-to-end tests for nano-eval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import hashlib
import json
from unittest.mock import patch

import pytest
import respx
from click.testing import CliRunner
from httpx import Response

from nano_eval import main
from tasks.chartqa import _score as chartqa_score, chartqa
from tasks.gsm8k import (
    _extract_gsm8k_answer,
    _normalize,
    _score as gsm8k_score,
    gsm8k_cot_llama,
)

# GSM8K: 10 mock responses keyed by prompt hash (8 correct, 2 wrong = 80% accuracy)
# Hashes are for the last user message (the question) in multiturn fewshot format
# fmt: off
GSM8K_RESPONSES = {
    "b4ae7b": "The final answer is $3",     # ✓ target=3 (dollar sign)
    "2d1210": "The final answer is 18.",    # ✓ target=18 (trailing period)
    "dbc356": "Hmm, 8 times 8 is 64",      # ✓ target=64 (fallback extraction, no "The final answer is")
    "a1f1f0": "The final answer is 20",     # ✓ target=20
    "050045": "The final answer is 999",    # ✗ target=45
    "464d52": "I first thought The final answer is 999, but I made an error. The final answer is 540",  # ✓ target=540 (last-match)
    "64816f": "The final answer is 999",    # ✗ target=160
    "967e59": "The final answer is $460",   # ✓ target=460 (dollar sign)
    "8c5053": "The final answer is 260",    # ✓ target=260
    "d85409": "The final answer is 70,000", # ✓ target=70000 (comma-formatted)
}
# fmt: on

# ChartQA: 10 mock responses keyed by prompt hash (8 correct, 2 wrong = 80% accuracy)
# fmt: off
CHARTQA_RESPONSES = {
    "eed350": "Final Answer: 6",      # ✓ target=6
    "c6ad9c": "**Final Answer:** 3",  # ✓ target=3 (markdown bold)
    "f9f65c": "Final Answer: No",     # ✓ target=No
    "fa51ed": "Final Answer: 14",     # ✓ target=14
    "8e6330": "Based on analysis, the answer: 62.",  # ✓ target=62 (trailing punct, non-standard prefix)
    "b2bd79": "Looking at the chart: FINAL ANSWER: 999. Wait, I need to recount... FINAL ANSWER: 23",  # ✓ target=23 (last-match)
    "372e0a": "FINAL ANSWER: wrong",  # ✗ target=Yes
    "833fc2": "Final Answer: 3%",     # ✓ target=0.03 (percent/decimal equivalence)
    "59f1fa": "Final Answer: 0.57",   # ✓ target=0.57
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


class TestGSM8KScoring:
    """Edge-case tests for GSM8K scoring against lm-eval reference behavior."""

    @pytest.mark.parametrize(
        "response, target, expected",
        [
            # Basic correct / incorrect
            ("The final answer is 42", "42", 1.0),
            ("The final answer is 99", "42", 0.0),
            # Negative numbers
            ("The final answer is -5", "-5", 1.0),
            ("The final answer is -100", "100", 0.0),
            # Decimals (string match, not numeric — "5.0" != "5")
            ("The final answer is 3.14", "3.14", 1.0),
            ("The final answer is 5.0", "5", 0.0),
            # Commas and dollar signs stripped by _normalize
            ("The final answer is 1,234", "1234", 1.0),
            ("The final answer is $50", "50", 1.0),
            ("The final answer is $1,234", "1234", 1.0),
            # Single-digit (must match via -?[0-9]+ alt in regex)
            ("The final answer is 3", "3", 1.0),
            # Self-correction: last "The final answer is" wins
            (
                "The final answer is 5. Wait, The final answer is 12",
                "12",
                1.0,
            ),
            (
                "The final answer is 5. Wait, The final answer is 12",
                "5",
                0.0,
            ),
            # Fallback: no "The final answer is", uses last number
            ("So the answer equals 42", "42", 1.0),
            ("I calculated 10 + 20 = 30 then 30 + 12 = 42", "42", 1.0),
            ("I calculated 10 + 20 = 30 then 30 + 12 = 42", "30", 0.0),
            # No numbers at all -> full response returned, won't match
            ("I don't know", "42", 0.0),
            # Trailing period: "42." captured by regex, stripped by _normalize
            ("The final answer is 42.", "42", 1.0),
            # Target with #### prefix (defensive normalization)
            ("The final answer is 42", "#### 42", 1.0),
            # Numbers in CoT before the final answer
            ("5 + 3 = 8. 8 * 2 = 16. The final answer is 16", "16", 1.0),
        ],
    )
    def test_score(self, response, target, expected):
        assert gsm8k_score(response, target) == expected

    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The final answer is 42", "42"),
            ("The final answer is -5", "-5"),
            ("The final answer is $1,234", "$1,234"),
            # Last match wins
            ("The final answer is 5. The final answer is 12", "12"),
            # Fallback to last number
            ("The answer is 42", "42"),
            ("10 + 20 = 30", "30"),
            # No numbers -> return full response
            ("no numbers here", "no numbers here"),
        ],
    )
    def test_extract_answer(self, response, expected):
        assert _extract_gsm8k_answer(response) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("42", "42"),
            ("$1,234", "1234"),
            ("#### 42", "42"),
            ("42.", "42"),
            ("HELLO", "hello"),
        ],
    )
    def test_normalize(self, text, expected):
        assert _normalize(text) == expected


class TestChartQAScoring:
    """Edge-case tests for ChartQA scoring against lm-eval reference behavior."""

    @pytest.mark.parametrize(
        "response, target, expected",
        [
            # Numeric exact match
            ("Final Answer: 42", "42", 1.0),
            # Within 5% tolerance: 105/100 = 5% relative change
            ("Final Answer: 105", "100", 1.0),
            # Beyond 5% tolerance: 106/100 = 6% relative change
            ("Final Answer: 106", "100", 0.0),
            # Boundary: target=200, pred=190 -> 10/200 = 5% exactly
            ("Final Answer: 190", "200", 1.0),
            # Just beyond: target=200, pred=189 -> 11/200 = 5.5%
            ("Final Answer: 189", "200", 0.0),
            # Target = 0: only pred ≈ 0 matches (denom becomes 1e-10)
            ("Final Answer: 0", "0", 1.0),
            ("Final Answer: 0.001", "0", 0.0),
            # Negative numbers
            ("Final Answer: -10", "-10", 1.0),
            ("Final Answer: -95", "-100", 1.0),  # 5% within
            # Percentage/decimal equivalence: "50%" and "0.5"
            ("Final Answer: 50%", "0.5", 1.0),
            ("Final Answer: 0.5", "50%", 1.0),
            ("Final Answer: 5%", "0.05", 1.0),
            # Same percentage: "50%" vs "50%"
            ("Final Answer: 50%", "50%", 1.0),
            # Dollar and commas in prediction
            ("Final Answer: $1,234", "1234", 1.0),
            # Text: case-insensitive exact match
            ("Final Answer: Yes", "Yes", 1.0),
            ("Final Answer: yes", "Yes", 1.0),
            ("Final Answer: YES", "Yes", 1.0),
            # Text: trailing punctuation stripped from prediction
            ("Final Answer: Yes.", "Yes", 1.0),
            # Text: wrong answer
            ("Final Answer: No", "Yes", 0.0),
            # No "answer:" marker -> score 0
            ("I think the result is 42", "42", 0.0),
            # Empty answer after "answer:" -> score 0
            ("Final Answer:", "42", 0.0),
            # Markdown formatting stripped: **42** -> 42
            ("Final Answer: **42**", "42", 1.0),
            # Self-correction: last "answer:" wins
            (
                "Answer: 999. Wait, let me recalculate. Answer: 42",
                "42",
                1.0,
            ),
            # Quoted target text
            ("Final Answer: hello", '"hello"', 1.0),
            # Multiline: answer on next line after "answer:"
            ("Final Answer:\n42", "42", 1.0),
        ],
    )
    def test_score(self, response, target, expected):
        assert chartqa_score(response, target) == expected


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
