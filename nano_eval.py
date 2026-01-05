"""
nano-eval - Minimal LLM evaluation tool.

Usage:
    # Python API
    from nano_eval import main
    result = main(tasks="gsm8k_cot_llama", base_url="http://localhost:8000/v1")

    # CLI
    nano-eval --tasks gsm8k_cot_llama --base_url http://localhost:8000/v1
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import click
import httpx

from core import APIConfig, LoggedSample, TaskResult, run_task
from tasks import TASKS

__all__ = ["main", "APIConfig", "run_task", "TASKS", "EvalResult"]


class ConfigInfo(TypedDict):
    model: str
    max_samples: int | None


class EvalResult(TypedDict):
    results: dict[str, TaskResult]
    eval_hash: str
    total_seconds: float
    config: ConfigInfo


def _parse_gen_kwargs(s: str | dict[str, Any]) -> dict[str, Any]:
    """Parse 'key=value,key=value' into dict, or pass through dict."""
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid format '{pair}': expected 'key=value'")
        key, value = pair.split("=", 1)
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
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
    with open(path / f"samples_{task_name}.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def _evaluate(
    task_names: list[str],
    config: APIConfig,
    max_samples: int | None,
    output_path: Path | None,
    log_samples: bool,
    seed: int | None,
) -> EvalResult:
    """Run evaluations (async implementation)."""
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


def main(
    tasks: str | list[str],
    base_url: str,
    *,
    model: str | None = None,
    api_key: str = "",
    num_concurrent: int = 8,
    max_retries: int = 3,
    gen_kwargs: str | dict[str, Any] = "",
    max_samples: int | None = None,
    output_path: str | Path | None = None,
    log_samples: bool = False,
    seed: int = 42,
) -> EvalResult:
    """
    Run LLM evaluations.

    Args:
        tasks: Task name(s) - comma-separated string or list
        base_url: API base URL (e.g. http://localhost:8000/v1)
        model: Model name (auto-detected if API serves only one)
        api_key: API authentication key
        num_concurrent: Max concurrent requests
        max_retries: Max retries per request
        gen_kwargs: Generation kwargs - "key=value,..." string or dict
        max_samples: Max samples per task (None for all)
        output_path: Directory for results.json and sample files
        log_samples: Write per-sample JSONL files
        seed: Seed for shuffling samples

    Returns:
        EvalResult with metrics, hashes, timing, and configuration
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    task_names = (
        [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else list(tasks)
    )
    base_url = base_url.rstrip("/")
    output_dir = Path(output_path) if output_path else None

    resolved_model = model
    if not resolved_model:
        models = _list_models(base_url, api_key)
        if len(models) == 1:
            resolved_model = models[0]
        else:
            raise ValueError(
                f"Auto-selecting model failed: expected 1 model, found {len(models)}. "
                "Please specify model."
            )

    default_gen_kwargs = {"temperature": 0, "max_tokens": 256, "seed": 42}
    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=resolved_model,
        api_key=api_key,
        num_concurrent=num_concurrent,
        max_retries=max_retries,
        gen_kwargs={**default_gen_kwargs, **_parse_gen_kwargs(gen_kwargs)},
    )

    return asyncio.run(
        _evaluate(task_names, config, max_samples, output_dir, log_samples, seed)
    )


@click.command()
@click.option(
    "--tasks",
    required=True,
    help=f"Comma-separated task names ({', '.join(TASKS.keys())})",
)
@click.option(
    "--base_url", required=True, help="API base URL (e.g. http://localhost:8000/v1)"
)
@click.option(
    "--model", default=None, help="Model name, auto-detected if API serves only one"
)
@click.option("--api_key", default="", help="API authentication key")
@click.option("--num_concurrent", default=8, type=int, help="Max concurrent requests")
@click.option("--max_retries", default=3, type=int, help="Max retries per request")
@click.option("--gen_kwargs", default="", help="Generation kwargs as key=value pairs")
@click.option("--max_samples", default=None, type=int, help="Max samples per task")
@click.option("--output_path", default=None, help="Directory for results.json")
@click.option("--log_samples", is_flag=True, help="Write per-sample JSONL files")
@click.option("--seed", default=42, type=int, help="Seed for shuffling samples")
def cli(**kwargs: Any) -> None:
    """nano-eval - Minimal LLM evaluation tool."""
    result = main(**kwargs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
