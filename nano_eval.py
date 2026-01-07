"""nano-eval: CLI for LLM evaluation on standardized tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import click
import httpx

__all__ = ["evaluate"]


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
    """Fetch available models from API /models endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [model["id"] for model in resp.json().get("data", [])]


async def evaluate(
    task_names: list[str],
    config,
    max_samples: int | None = None,
    output_path: Path | None = None,
    log_samples: bool = False,
    seed: int | None = None,
) -> dict:
    """Run evaluations for specified tasks."""
    from core import run_task
    from tasks import TASKS

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    task_hashes = []
    total_seconds = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await run_task(TASKS[name], config, max_samples, seed)
        task_hashes.append(result["task_hash"])
        if output_path and log_samples:
            filepath = output_path / f"samples_{name}.jsonl"
            with open(filepath, "w") as f:
                for sample in result["samples"]:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        results[name] = {
            "task": result["task"],
            "task_hash": result["task_hash"],
            "metrics": result["metrics"],
            "num_samples": result["num_samples"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        total_seconds += result["elapsed_seconds"]

    eval_result = {
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
@click.option(
    "-t",
    "--tasks",
    type=click.Choice(["gsm8k_cot_llama", "chartqa"]),
    required=True,
    multiple=True,
    help="Task to evaluate",
)
@click.option("--base-url", required=True, help="OpenAI-compatible API endpoint")
@click.option("--model", help="Model name (auto-detected if single model)")
@click.option("--api-key", default="", help="API authentication token")
@click.option(
    "--num-concurrent", default=8, show_default=True, help="Parallel requests"
)
@click.option("--max-retries", default=3, show_default=True, help="Retry attempts")
@click.option(
    "--extra-request-params",
    "gen_kwargs",
    default="temperature=0,max_tokens=256,seed=42",
    show_default=True,
    help="API params as key=value,...",
)
@click.option("--max-samples", type=int, help="Limit samples per task")
@click.option("--output-path", type=click.Path(), help="Output directory for results")
@click.option("--log-samples", is_flag=True, help="Save per-sample JSONL")
@click.option("--seed", default=42, show_default=True, help="Shuffle seed")
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
    """Evaluate LLMs on tasks via OpenAI-compatible APIs."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    from core import APIConfig

    base_url = base_url.rstrip("/")
    if not model:
        models = _list_models(base_url, api_key)
        if len(models) != 1:
            raise click.UsageError(
                f"Auto-detect failed: found {len(models)} models. Specify --model."
            )
        model = models[0]

    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=api_key,
        num_concurrent=num_concurrent,
        max_retries=max_retries,
        gen_kwargs=_parse_kwargs(gen_kwargs),
    )

    path = Path(output_path) if output_path else None
    output = asyncio.run(
        evaluate(list(tasks), config, max_samples, path, log_samples, seed)
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
