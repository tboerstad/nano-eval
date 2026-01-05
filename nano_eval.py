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

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict

import httpx

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
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Failed to fetch models from {url}: HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to connect to {url}: {e}") from e
    data = resp.json()
    if "data" not in data:
        raise RuntimeError(f"Unexpected response from {url}: missing 'data' field")
    models = []
    for model in data["data"]:
        if "id" not in model:
            continue
        models.append(model["id"])
    return models


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


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="nano-eval - Minimal LLM evaluation (chat/completions models only)"
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help=f"Comma-separated task names (available: {', '.join(TASKS.keys())})",
    )
    parser.add_argument(
        "--base_url",
        required=True,
        help="API base URL (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        help="Model name, auto-detected if API serves only one (e.g. google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--api_key", default="", help="API authentication key (default: none)"
    )
    parser.add_argument(
        "--num_concurrent",
        type=int,
        default=8,
        help="Max concurrent requests (default: 8)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries per request (default: 3)",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help="Generation kwargs as key=value pairs. "
        "Defaults: temperature=0,max_tokens=256,seed=42. "
        "User values override defaults (e.g. max_tokens=1024 keeps temperature=0,seed=42)",
    )
    parser.add_argument(
        "--max_samples", type=int, help="Max samples per task (default: all)"
    )
    parser.add_argument(
        "--output_path",
        help="Directory for results.json and sample files (e.g. ./results)",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Write per-sample JSONL files (default: false)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffling samples (default: 42)",
    )
    args = parser.parse_args()

    if args.num_concurrent < 1:
        parser.error("--num_concurrent must be at least 1")
    if args.max_retries < 1:
        parser.error("--max_retries must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max_samples must be at least 1")

    base_url = args.base_url.rstrip("/")
    model = args.model

    if not model:
        models = _list_models(base_url, args.api_key)
        if len(models) == 1:
            model = models[0]
        else:
            parser.error(
                f"Auto-selecting model failed: expected 1 model, found {len(models)}. "
                "Please specify --model."
            )

    default_gen_kwargs = {"temperature": 0, "max_tokens": 256, "seed": 42}
    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=args.api_key,
        num_concurrent=args.num_concurrent,
        max_retries=args.max_retries,
        gen_kwargs={**default_gen_kwargs, **_parse_kwargs(args.gen_kwargs)},
    )

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_path = Path(args.output_path) if args.output_path else None
    output = asyncio.run(
        evaluate(
            task_names,
            config,
            args.max_samples,
            output_path,
            args.log_samples,
            args.seed,
        )
    )

    print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
