"""nano-eval: evaluate LLMs on standardized tasks via OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import click
import httpx

if TYPE_CHECKING:
    from core import TaskResult

logger = logging.getLogger("nano_eval")


class EvalResult(TypedDict):
    config: dict[str, Any]
    framework_version: str
    results: dict[str, TaskResult]
    total_seconds: float


class _LevelPrefixFormatter(logging.Formatter):
    """Prefixes WARNING/ERROR messages with their level, passes INFO through clean."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {record.getMessage()}"
        return record.getMessage()


def _parse_kwargs(s: str) -> dict[str, Any]:
    """Parse 'key=value,key=value' into dict."""
    if not s:
        return {}
    result: dict[str, Any] = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid format '{pair}': expected 'key=value'")
        key, value = pair.split("=", 1)
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def _check_endpoint(url: str, api_key: str = "") -> None:
    """Verify API endpoint is reachable."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code in (401, 403):
            raise ValueError(
                f"Authentication failed ({resp.status_code}) at {url}\n"
                "Check your --api-key value."
            )
        if resp.status_code not in (404, 405):
            resp.raise_for_status()
    except httpx.HTTPError:
        raise ValueError(f"No response from {url}\nIs the server running?")


def _detect_base_url(api_key: str = "") -> str:
    """Try common local ports to find an API server."""
    candidates = [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8080/v1",
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for base_url in candidates:
        try:
            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=2)
            if resp.status_code in (200, 401):
                return base_url
        except httpx.HTTPError:
            continue
    raise ValueError(
        f"No local API server found. Tried: {', '.join(candidates)}\n"
        "Start a server or provide --base-url explicitly."
    )


def _list_models(base_url: str, api_key: str = "") -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{base_url}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [model["id"] for model in resp.json().get("data", [])]


def evaluate(
    modalities: list[str],
    base_url: str | None = None,
    model: str | None = None,
    api_key: str = "",
    max_concurrent: int = 8,
    gen_kwargs: dict[str, Any] | None = None,
    max_samples: int | None = None,
    output_path: Path | None = None,
    log_requests: bool = False,
    dataset_seed: int = 42,
    request_timeout: int = 300,
) -> EvalResult:
    """Run evaluations for specified modalities and return results dict."""
    from core import ApiConfig, run_task
    from tasks import TASKS

    if base_url is None:
        base_url = _detect_base_url(api_key)
        logger.info(
            f"`base_url` not provided, using auto-detected endpoint: {base_url}"
        )

    base_url = base_url.rstrip("/")
    logger.info(f"Checking that endpoint is responding: {base_url}/chat/completions")
    _check_endpoint(f"{base_url}/chat/completions", api_key)

    if model is None:
        models = _list_models(base_url, api_key)
        if len(models) == 1:
            model = models[0]
            logger.info(f"`model` not provided, using auto-detected model: {model}")
        else:
            raise ValueError(
                f"Auto-detecting model failed: found {len(models)} models: {', '.join(models)}. "
                "Please specify model explicitly."
            )

    config = ApiConfig(
        url=f"{base_url}/chat/completions",
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        timeout=request_timeout,
        gen_kwargs=gen_kwargs or {},
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, TaskResult] = {}
    total_seconds = 0.0

    for modality in modalities:
        if modality not in TASKS:
            raise ValueError(
                f"Unknown modality: {modality}. Available: {list(TASKS.keys())}"
            )
        task = TASKS[modality]
        result, request_logs = asyncio.run(
            run_task(task, config, modality, max_samples, dataset_seed)
        )
        if output_path and log_requests:
            requests_file = output_path / f"request_log_{modality}.jsonl"
            with open(requests_file, "w") as f:
                for entry in request_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(
                f"Request logs for {modality} dataset written to: {requests_file}"
            )
        results[modality] = result
        total_seconds += result["elapsed_seconds"]

    eval_result = EvalResult(
        config={"max_samples": max_samples, "model": config.model},
        framework_version=version("nano-eval"),
        results=results,
        total_seconds=total_seconds,
    )

    if output_path:
        results_file = output_path / "eval_results.json"
        with open(results_file, "w") as f:
            json.dump(eval_result, f, indent=2)
        logger.info(f"Evaluation result written to: {results_file}")

    return eval_result


def _print_results_table(result: EvalResult) -> None:
    print("\nTask    Accuracy  Samples  Duration  Output Tokens  Per Req Tok/s")
    print("------  --------  -------  --------  -------------  -------------")
    for r in result["results"].values():
        print(
            f"{r['modality']:<6}  {r['metrics']['accuracy']:>7.1%}  {r['num_samples']:>7}  {int(r['elapsed_seconds']):>7}s  {r['total_output_tokens']:>13}  {int(r['tokens_per_second']):>13}"
        )


@click.command()
@click.option(
    "-m",
    "--modality",
    "modalities",
    type=click.Choice(["text", "vision"]),
    required=True,
    multiple=True,
    help="Modality to evaluate (can be repeated)",
)
@click.option(
    "--base-url",
    help="OpenAI-compatible API endpoint; tries 127.0.0.1:8000/8080 if omitted",
)
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
    help="Write eval_results.json and request logs to this directory",
)
@click.option(
    "--log-requests",
    is_flag=True,
    help="Save per-request results as JSONL (requires --output-path)",
)
@click.option(
    "--dataset-seed", default=42, show_default=True, help="Controls sample order"
)
@click.option(
    "--request-timeout",
    default=300,
    show_default=True,
    help="Timeout in seconds for each API request",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (up to -vvv)",
)
@click.version_option(version=version("nano-eval"), prog_name="nano-eval")
def main(
    modalities: tuple[str, ...],
    base_url: str | None,
    model: str | None,
    api_key: str,
    max_concurrent: int,
    gen_kwargs: str,
    max_samples: int | None,
    output_path: str | None,
    log_requests: bool,
    dataset_seed: int,
    request_timeout: int,
    verbose: int,
) -> None:
    """Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

    Example: nano-eval -m text
    """
    if verbose < 1:
        handler = logging.StreamHandler()
        handler.setFormatter(_LevelPrefixFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    if verbose >= 1:
        logger.setLevel(logging.DEBUG)
    if verbose >= 2:
        logging.getLogger("httpx").setLevel(logging.INFO)
    if verbose >= 3:
        logging.getLogger("httpx").setLevel(logging.DEBUG)

    result = evaluate(
        modalities=list(modalities),
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        gen_kwargs=_parse_kwargs(gen_kwargs),
        max_samples=max_samples,
        output_path=Path(output_path) if output_path else None,
        log_requests=log_requests,
        dataset_seed=dataset_seed,
        request_timeout=request_timeout,
    )

    _print_results_table(result)


if __name__ == "__main__":
    main()
