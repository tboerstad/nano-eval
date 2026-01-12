"""
Tests for nano-eval CLI.
"""


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
