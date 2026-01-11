"""
nano-eval CLI entry point.

CLI args → APIConfig → evaluate() → TASKS[type]() → JSON output
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click
import httpx

from schema import (
    ConfigInfo,
    EvalResult,
    LoggedSample,
    Metrics,
    TaskResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate",
    "EvalResult",
    "ConfigInfo",
    "TaskResult",
    "Metrics",
    "LoggedSample",
]


def _parse_kwargs(s: str) -> dict[str, str | int | float]:
    """Parse 'key=value,key=value' into dict."""
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid format '{pair}': expected 'key=value'")
        key, value = pair.split("=", 1)
        try:
            # Parse numbers/bools: temperature=0.7 -> 0.7, enabled=true -> True
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            # Unquoted strings fail JSON parsing, use as-is: model=gpt-4 -> "gpt-4"
            result[key] = value
    return result


def _check_endpoint(url: str, api_key: str = "") -> None:
    """Verify API endpoint is reachable. Raises ValueError with user-friendly message."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        # 404/405 mean server is running (GET not supported on chat/completions)
        if resp.status_code not in (404, 405):
            resp.raise_for_status()
    except httpx.HTTPError:
        raise ValueError(f"No response from {url}\nIs the server running?")


def _list_models(base_url: str, api_key: str = "") -> list[str]:
    """Fetch available models from the API's /models endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [model["id"] for model in resp.json().get("data", [])]


def _write_samples_jsonl(filepath: Path, samples: list[LoggedSample]) -> None:
    """Write per-sample results to JSONL file."""
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def evaluate(
    types: list[str],
    base_url: str,
    model: str | None = None,
    api_key: str = "",
    max_concurrent: int = 8,
    gen_kwargs: dict[str, Any] | None = None,
    max_samples: int | None = None,
    output_path: Path | None = None,
    log_samples: bool = False,
    seed: int = 42,
) -> EvalResult:
    """
    Run evaluations for specified types.

    Args:
        types: List of types to evaluate ("text" or "vision")
        base_url: OpenAI-compatible API endpoint (e.g. http://localhost:8000/v1)
        model: Model name. Auto-detected if endpoint serves exactly one model.
        api_key: Bearer token for API authentication
        gen_kwargs: API params like temperature, max_tokens, seed
        max_samples: Optional limit on samples per task
        output_path: If provided, write results.json to this directory
        log_samples: If True, also write samples_{task}.jsonl files

    Returns:
        EvalResult with per-task metrics and metadata
    """
    from core import APIConfig, run_task
    from tasks import TASKS

    base_url = base_url.rstrip("/")
    logger.info(f"Checking that endpoint is responding: {base_url}/chat/completions")
    _check_endpoint(f"{base_url}/chat/completions", api_key)

    if model is None:
        models = _list_models(base_url, api_key)
        if len(models) == 1:
            model = models[0]
            logger.info(f"Successfully auto-detected model: {model}")
        else:
            raise ValueError(
                f"Auto-detecting model failed: found {len(models)} models: {', '.join(models)}. "
                "Please specify model explicitly."
            )

    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        gen_kwargs=gen_kwargs or {},
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, TaskResult] = {}
    total_seconds = 0.0

    for type_name in types:
        if type_name not in TASKS:
            raise ValueError(
                f"Unknown type: {type_name}. Available: {list(TASKS.keys())}"
            )
        task = TASKS[type_name]
        result = await run_task(task, config, max_samples, seed)
        if output_path and log_samples:
            samples_file = output_path / f"samples_{task.name}.jsonl"
            _write_samples_jsonl(samples_file, result["samples"])
            logger.info(f"`{type_name.title()}` sample log written to: {samples_file}")
        results[type_name] = TaskResult(
            elapsed_seconds=result["elapsed_seconds"],
            metrics=result["metrics"],
            num_samples=result["num_samples"],
            samples_hash=result["samples_hash"],
            task=result["task"],
            task_type=result["task_type"],
        )
        total_seconds += result["elapsed_seconds"]

    eval_result: EvalResult = {
        "config": {"max_samples": max_samples, "model": config.model},
        "framework_version": version("nano-eval"),
        "results": results,
        "total_seconds": round(total_seconds, 2),
    }

    if output_path:
        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(eval_result, f, indent=2)
        logger.info(f"Results written to: {results_file}")

    return eval_result


def _print_results_table(result: EvalResult) -> None:
    """Print a mini results table."""
    print("\nTask    Accuracy  Samples  Duration")
    print("------  --------  -------  --------")
    for r in result["results"].values():
        print(
            f"{r['task_type']:<6}  {r['metrics']['exact_match']:>7.1%}  {r['num_samples']:>7}  {int(r['elapsed_seconds']):>7}s"
        )


@click.command()
@click.option(
    "-t",
    "--type",
    "types",
    type=click.Choice(["text", "vision"]),
    required=True,
    multiple=True,
    help="Type to evaluate (can be repeated)",
)
@click.option("--base-url", required=True, help="OpenAI-compatible API endpoint")
@click.option("--model", help="Model name; auto-detected if endpoint serves one model")
@click.option("--api-key", default="", help="Bearer token for API authentication")
@click.option("--max-concurrent", default=8, show_default=True)
@click.option(
    "--extra-request-params",
    "gen_kwargs",
    default="temperature=0,max_tokens=256,seed=42",
    show_default=True,
    help="API params as key=value,...",
)
@click.option("--max-samples", type=int, help="If provided, limit samples per task")
@click.option(
    "--output-path",
    type=click.Path(),
    help="Write results.json and sample logs to this directory",
)
@click.option(
    "--log-samples",
    is_flag=True,
    help="Save per-sample results as JSONL (requires --output-path)",
)
@click.option("--seed", default=42, show_default=True, help="Controls sample order")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (up to -vv)",
)
@click.version_option(version=version("nano-eval"), prog_name="nano-eval")
def main(
    types: tuple[str, ...],
    base_url: str,
    model: str | None,
    api_key: str,
    max_concurrent: int,
    gen_kwargs: str,
    max_samples: int | None,
    output_path: str | None,
    log_samples: bool,
    seed: int,
    verbose: int,
) -> None:
    """Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

    Example: nano-eval -t text --base-url http://localhost:8000/v1
    """
    log_level = logging.DEBUG if verbose >= 2 else logging.INFO
    log_format = "%(message)s" if verbose < 1 else logging.BASIC_FORMAT
    logging.basicConfig(level=log_level, format=log_format)
    if verbose < 1:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    result = asyncio.run(
        evaluate(
            types=list(types),
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_concurrent=max_concurrent,
            gen_kwargs=_parse_kwargs(gen_kwargs),
            max_samples=max_samples,
            output_path=Path(output_path) if output_path else None,
            log_samples=log_samples,
            seed=seed,
        )
    )

    _print_results_table(result)


if __name__ == "__main__":
    main()
