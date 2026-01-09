"""
End-to-end tests for nano-eval CLI.

Tests use real tasks with mocked API calls via complete().
"""

import hashlib
import json
from unittest.mock import patch

from click.testing import CliRunner

from nano_eval import main
from tasks.chartqa import samples as chartqa_samples
from tasks.gsm8k import samples as gsm8k_samples

# Pre-load samples once at module level
_GSM8K_SAMPLES = gsm8k_samples(10, 42)
_CHARTQA_SAMPLES = chartqa_samples(10, 42)

# Build response dicts (7 correct, 3 wrong = 70% accuracy)
GSM8K_RESPONSES: dict[str, str] = {}
for i, s in enumerate(_GSM8K_SAMPLES):
    prompt = s.prompt
    assert isinstance(prompt, list)  # multiturn messages
    last_user = [m for m in prompt if m["role"] == "user"][-1]["content"]
    h = hashlib.md5(last_user.encode()).hexdigest()[:6]
    answer = "999" if i in (4, 5, 6) else s.target
    GSM8K_RESPONSES[h] = f"The final answer is {answer}"

CHARTQA_RESPONSES: dict[str, str] = {}
for i, s in enumerate(_CHARTQA_SAMPLES):
    prompt = s.prompt
    assert isinstance(prompt, tuple)  # (text, images)
    h = hashlib.md5(prompt[0].encode()).hexdigest()[:6]
    answer = "999" if i in (4, 5, 6) else s.target
    CHARTQA_RESPONSES[h] = f"FINAL ANSWER: {answer}"


class TestE2E:
    """End-to-end tests with real tasks and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with mocked API."""

        async def mock_complete(prompts, config):
            responses = []
            for prompt in prompts:
                last_user = [m for m in prompt if m["role"] == "user"][-1]["content"]
                h = hashlib.md5(last_user.encode()).hexdigest()[:6]
                responses.append(GSM8K_RESPONSES[h])
            return responses

        with patch("core.complete", new=mock_complete):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--tasks=gsm8k_cot_llama",
                    "--base-url=http://test.com/v1",
                    "--model=test",
                    "--max-samples=10",
                    "--output-path",
                    str(tmp_path),
                    "--log-samples",
                ],
            )
            assert result.exit_code == 0, result.output

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["gsm8k_cot_llama"]["metrics"]["exact_match"] == 0.7

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with mocked API."""

        async def mock_complete(prompts, config):
            responses = []
            for prompt in prompts:
                text = prompt[0]  # (text, images) tuple
                h = hashlib.md5(text.encode()).hexdigest()[:6]
                responses.append(CHARTQA_RESPONSES[h])
            return responses

        with patch("core.complete", new=mock_complete):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--tasks=chartqa",
                    "--base-url=http://test.com/v1",
                    "--model=test",
                    "--max-samples=10",
                    "--output-path",
                    str(tmp_path),
                    "--log-samples",
                ],
            )
            assert result.exit_code == 0, result.output

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["chartqa"]["metrics"]["exact_match"] == 0.7
