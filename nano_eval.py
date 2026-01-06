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
from typing import TYPE_CHECKING, TypedDict

import click
import httpx

if TYPE_CHECKING:
    from core import APIConfig, LoggedSample, TaskResult, run_task
    from tasks import TASKS

__all__ = ["APIConfig", "run_task", "TASKS", "evaluate"]


class ConfigInfo(TypedDict):
    model: str
    max_samples: int | None


class EvalResult(TypedDict):
    results: dict[str, TaskResult]
    eval_hash: str
    total_seconds: float
    config: ConfigInfo


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


@click.command()
@click.option("--tasks", "-t", type=click.Choice(["gsm8k_cot_llama", "chartqa"]), required=True, multiple=True)
@click.option("--base-url", required=True, help="API base URL")
@click.option("--model", help="Auto-detected if API serves only one")
@click.option("--api-key", default="")
@click.option("--num-concurrent", default=8, show_default=True)
@click.option("--max-retries", default=3, show_default=True)
@click.option("--gen-kwargs", default="", help="key=value,... [temperature=0,max_tokens=256,seed=42]")
@click.option("--max-samples", type=int)
@click.option("--output-path", type=click.Path(), help="Output directory")
@click.option("--log-samples", is_flag=True, help="Write per-sample JSONL")
@click.option("--seed", default=42, show_default=True)
def main(
    tasks: tuple[str, ...],
    base_url: str,
    model: str | None,
    api_key: str,
    num_concurrent: int,
    max_retries: int,
    gen_kwargs: str,
    max_samples: int | None,
    output_path: str | None,
    log_samples: bool,
    seed: int,
) -> None:
    """nano-eval - LLM evaluation for chat/completions models."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from core import APIConfig

    base_url = base_url.rstrip("/")

    if not model:
        models = _list_models(base_url, api_key)
        if len(models) == 1:
            model = models[0]
        else:
            raise click.UsageError(
                f"Auto-selecting model failed: expected 1 model, found {len(models)}. "
                "Please specify --model."
            )

    default_gen_kwargs = {"temperature": 0, "max_tokens": 256, "seed": 42}
    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=api_key,
        num_concurrent=num_concurrent,
        max_retries=max_retries,
        gen_kwargs={**default_gen_kwargs, **_parse_kwargs(gen_kwargs)},
    )

    task_names = list(tasks)
    path = Path(output_path) if output_path else None
    output = asyncio.run(
        evaluate(task_names, config, max_samples, path, log_samples, seed)
    )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
