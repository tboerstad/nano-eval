"""
nano-eval CLI entry point.

Responsibilities:
- Parse CLI args (tasks, model, endpoint, concurrency)
- Create APIConfig, run tasks
- Output JSON

Architecture:
    nano_eval.py (CLI, orchestration)
         │
    ┌────┴────┐
  core.py   tasks/
  APIConfig   TASKS registry
  complete()  gsm8k.py
  run_task()  chartqa.py

Flow: CLI → APIConfig → evaluate() → TASKS[name]() → JSON
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import click
import httpx

if TYPE_CHECKING:
    from core import APIConfig, LoggedSample, TaskResult, run_task
    from tasks import TASKS

JsonValue: TypeAlias = (
    str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
)

__all__ = ["run_eval", "EvalResult", "APIConfig", "run_task", "TASKS", "evaluate"]


class ConfigInfo(TypedDict):
    model: str
    max_samples: int | None


class EvalResult(TypedDict):
    results: dict[str, TaskResult]
    eval_hash: str
    total_seconds: float
    config: ConfigInfo


def _parse_kwargs(s: str) -> dict[str, JsonValue]:
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


def _list_models(base_url: str, api_key: str = "") -> list[str]:
    """Fetch available models from the API's /models endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [model["id"] for model in resp.json().get("data", [])]


def _write_samples_jsonl(
    path: Path, task_name: str, samples: list[LoggedSample]
) -> None:
    """Write per-sample results to JSONL file."""
    filepath = path / f"samples_{task_name}.jsonl"
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def evaluate(
    task_names: list[str],
    config: APIConfig,
    max_samples: int | None = None,
    output_path: Path | None = None,
    log_samples: bool = False,
    seed: int | None = None,
) -> EvalResult:
    """
    Run evaluations for specified tasks.

    Args:
        task_names: List of task names to evaluate
        config: API configuration
        max_samples: Optional limit on samples per task
        output_path: If provided, write results.json to this directory
        log_samples: If True, also write samples_{task}.jsonl files
        seed: Optional seed for shuffling samples
    """
    from core import TaskResult, run_task
    from tasks import TASKS

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, TaskResult] = {}
    task_hashes: list[str] = []
    total_seconds = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await run_task(TASKS[name], config, max_samples, seed)
        task_hashes.append(result["task_hash"])
        if output_path and log_samples:
            _write_samples_jsonl(output_path, name, result["samples"])
        results[name] = TaskResult(
            task=result["task"],
            task_hash=result["task_hash"],
            metrics=result["metrics"],
            num_samples=result["num_samples"],
            elapsed_seconds=result["elapsed_seconds"],
        )
        total_seconds += result["elapsed_seconds"]

    eval_result: EvalResult = {
        "results": results,
        "eval_hash": hashlib.sha256("".join(sorted(task_hashes)).encode()).hexdigest(),
        "total_seconds": round(total_seconds, 2),
        "config": {"model": config.model, "max_samples": max_samples},
    }

    if output_path:
        with open(output_path / "results.json", "w") as f:
            json.dump(eval_result, f, indent=2)

    return eval_result


DEFAULT_GEN_KWARGS: dict[str, JsonValue] = {
    "temperature": 0,
    "max_tokens": 256,
    "seed": 42,
}


def run_eval(
    tasks: list[str],
    base_url: str,
    model: str | None = None,
    api_key: str = "",
    num_concurrent: int = 8,
    max_retries: int = 3,
    gen_kwargs: dict[str, JsonValue] | None = None,
    max_samples: int | None = None,
    output_path: Path | str | None = None,
    log_samples: bool = False,
    seed: int = 42,
) -> EvalResult:
    """
    Run evaluations on specified tasks.

    This is the main Python API. The CLI forwards to this function.

    Args:
        tasks: Task names to evaluate (e.g., ["gsm8k_cot_llama", "chartqa"])
        base_url: OpenAI-compatible API endpoint
        model: Model name; auto-detected if endpoint serves one model
        api_key: Bearer token for API authentication
        num_concurrent: Parallel requests to send
        max_retries: Retry attempts for failed requests
        gen_kwargs: API params (temperature, max_tokens, seed). Defaults to
            {"temperature": 0, "max_tokens": 256, "seed": 42}
        max_samples: Limit samples per task
        output_path: Write results.json and sample logs to this directory
        log_samples: Save per-sample results as JSONL (requires output_path)
        seed: Seed for shuffling samples

    Returns:
        EvalResult with results, eval_hash, timing, and config
    """
    from core import APIConfig

    base_url = base_url.rstrip("/")

    if not model:
        models = _list_models(base_url, api_key)
        if len(models) == 1:
            model = models[0]
        else:
            raise ValueError(
                f"Auto-selecting model failed: expected 1 model, found {len(models)}. "
                "Please specify model."
            )

    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=api_key,
        num_concurrent=num_concurrent,
        max_retries=max_retries,
        gen_kwargs=gen_kwargs if gen_kwargs is not None else DEFAULT_GEN_KWARGS,
    )

    path = Path(output_path) if output_path else None
    return asyncio.run(evaluate(tasks, config, max_samples, path, log_samples, seed))


@click.command()
@click.option(
    "--tasks", "-t", multiple=True, required=True, help="Task to evaluate (repeatable)"
)
@click.option("--base-url", required=True, help="OpenAI-compatible API endpoint")
@click.option(
    "--model", default=None, help="Model name (auto-detected if endpoint serves one)"
)
@click.option("--api-key", default="", help="Bearer token for API authentication")
@click.option(
    "--num-concurrent", default=8, show_default=True, help="Parallel requests"
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    help="Retry attempts for failed requests",
)
@click.option(
    "--extra-request-params",
    default="temperature=0,max_tokens=256,seed=42",
    show_default=True,
    help="API params as key=value,...",
)
@click.option("--max-samples", default=None, type=int, help="Limit samples per task")
@click.option(
    "--output-path", default=None, type=click.Path(), help="Directory for results.json"
)
@click.option("--log-samples", is_flag=True, help="Save per-sample JSONL")
@click.option(
    "--seed", default=42, show_default=True, help="Seed for shuffling samples"
)
def main(
    tasks: tuple[str, ...],
    base_url: str,
    model: str | None,
    api_key: str,
    num_concurrent: int,
    max_retries: int,
    extra_request_params: str,
    max_samples: int | None,
    output_path: str | None,
    log_samples: bool,
    seed: int,
) -> None:
    """Evaluate LLMs on standardized tasks via OpenAI-compatible APIs."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        result = run_eval(
            tasks=list(tasks),
            base_url=base_url,
            model=model,
            api_key=api_key,
            num_concurrent=num_concurrent,
            max_retries=max_retries,
            gen_kwargs=_parse_kwargs(extra_request_params),
            max_samples=max_samples,
            output_path=output_path,
            log_samples=log_samples,
            seed=seed,
        )
    except ValueError as e:
        if "Auto-selecting model" in str(e):
            raise click.UsageError(str(e).replace("model.", "--model."))
        raise

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
