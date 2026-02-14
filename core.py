"""Core data types and async evaluation engine."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import datasets
import datasets.config as ds_config
import httpx
from datasets import Dataset, DownloadMode
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("nano_eval.core")


class Metrics(TypedDict):
    accuracy: float
    accuracy_stderr: float


class TaskResult(TypedDict):
    elapsed_seconds: float
    metrics: Metrics
    num_samples: int
    samples_hash: str
    task: str
    modality: str
    total_input_tokens: int
    total_output_tokens: int
    tokens_per_second: float


@dataclass(frozen=True)
class Prompt:
    text: str | list[dict[str, str]]
    images: list[Any] = field(default_factory=list)


@dataclass
class Sample:
    prompt: Prompt
    target: str


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    revision: str
    splits: list[str]
    extract: Callable[[dict[str, Any]], Sample]
    score: Callable[[str, str], float]
    config_name: str | None = None

    def load_samples(
        self, max_samples: int | None = None, seed: int | None = None
    ) -> list[Sample]:
        """Load samples from HuggingFace dataset across splits."""
        # TODO Upstream fix. HF datasets logging is too noisy
        datasets.utils.logging.set_verbosity_error()

        cache_path = (
            Path(HF_HUB_CACHE)
            / f"datasets--{self.dataset.replace('/', '--')}"
            / "snapshots"
            / self.revision
        )
        cached = cache_path.is_dir()
        logger.info(
            f"Cache {'hit' if cached else 'miss'} for {self.dataset}, HF_HOME={HF_HOME}"
        )
        old_offline = ds_config.HF_HUB_OFFLINE
        if cached:
            ds_config.HF_HUB_OFFLINE = True

        try:
            result: list[Sample] = []
            for split in self.splits:
                remaining = None if max_samples is None else max_samples - len(result)
                if remaining is not None and remaining <= 0:
                    break
                ds = datasets.load_dataset(
                    self.dataset,
                    name=self.config_name,
                    split=split,
                    revision=self.revision,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                )
                assert isinstance(ds, Dataset)
                if seed is not None:
                    ds = ds.shuffle(seed=seed)
                if remaining is not None:
                    ds = ds.select(range(min(remaining, len(ds))))
                for doc in ds:
                    result.append(self.extract(doc))
            return result
        finally:
            ds_config.HF_HUB_OFFLINE = old_offline


@dataclass
class ApiConfig:
    url: str
    model: str
    api_key: str = ""
    max_concurrent: int = 8
    timeout: int = 300
    gen_kwargs: dict[str, Any] = field(default_factory=dict)


async def _request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload)
    if resp.is_success:
        data = resp.json()
        return {
            "answer": data["choices"][0]["message"]["content"],
            "stop_reason": data["choices"][0]["finish_reason"],
            "input_tokens": data["usage"]["prompt_tokens"],
            "output_tokens": data["usage"]["completion_tokens"],
            "duration_seconds": time.perf_counter() - t0,
        }
    raise RuntimeError(f"Request failed: {resp.text}")


async def complete(
    prompts: list[Prompt],
    config: ApiConfig,
    progress_desc: str = "Running evals",
) -> list[dict[str, Any]]:
    """Run batch of chat completions with concurrency control."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.max_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for prompt in prompts:
            if prompt.images:
                assert isinstance(prompt.text, str)
                messages = _build_vision_message(prompt.text, prompt.images)
            elif isinstance(prompt.text, list):
                messages = prompt.text
            else:
                messages = [{"role": "user", "content": prompt.text}]

            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                **config.gen_kwargs,
            }

            tasks.append(asyncio.create_task(_request(client, config.url, payload)))

        try:
            return list(
                await tqdm_asyncio.gather(*tasks, desc=progress_desc, leave=False)
            )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def _build_vision_message(text: str, images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for img in images:
        if b64 := _encode_image(img):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    content.append({"type": "text", "text": text.strip()})
    return [{"role": "user", "content": content}]


def _encode_image(image: Any) -> str:
    """Encode PIL image to base64, or pass through base64 string."""
    if isinstance(image, str):
        if image.startswith("http"):
            raise ValueError("Remote image URLs are not supported.")
        return image

    if isinstance(image, Image.Image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def _prompt_text(prompt: Prompt) -> str:
    if isinstance(prompt.text, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in prompt.text)
    return prompt.text


def compute_samples_hash(samples: list[Sample]) -> str:
    """SHA256 hash of all samples for reproducibility tracking."""
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(_prompt_text(s.prompt).encode())
        for img in s.prompt.images:
            hasher.update(_encode_image(img).encode())
        hasher.update(s.target.encode())
    return hasher.hexdigest()


async def run_task(
    task: Task,
    config: ApiConfig,
    modality: str,
    max_samples: int | None = None,
    seed: int | None = None,
) -> tuple[TaskResult, list[dict[str, Any]]]:
    """Evaluate a task: load samples, run inference, compute scores."""
    samples = task.load_samples(max_samples, seed)
    samples_hash = compute_samples_hash(samples)
    prompts = [s.prompt for s in samples]

    logger.info(
        f"Starting {modality} ({task.name}) eval: "
        f"{len(samples)} samples, up to {config.max_concurrent} concurrent requests"
    )
    t0 = time.perf_counter()
    responses = await complete(prompts, config, f"Running {modality} eval")
    elapsed = time.perf_counter() - t0

    reason_counts = Counter(r["stop_reason"] for r in responses)
    non_stop = sum(c for reason, c in reason_counts.items() if reason != "stop")
    if non_stop:
        logger.warning(
            f"{non_stop} response(s) did not finish with 'stop'. "
            f"Reasons: {dict(reason_counts)}"
        )

    scores = [task.score(r["answer"], s.target) for r, s in zip(responses, samples)]
    n = len(samples)

    if logger.isEnabledFor(logging.DEBUG):
        for i, (s, r, score) in enumerate(zip(samples, responses, scores)):
            status = "PASS" if score == 1.0 else "FAIL"
            logger.debug(
                f"[{i + 1}/{n} {status}] target={s.target!r} got={r['answer']!r}"
            )

    accuracy = sum(scores) / n if n else 0.0
    stderr = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0
    logger.debug(
        f"{task.name}: accuracy={accuracy:.4f}+/-{stderr:.4f} ({elapsed:.2f}s)"
    )

    request_logs = [
        {
            "request_id": i,
            "target": s.target,
            "prompt": _prompt_text(s.prompt),
            "response": r["answer"],
            "score": score,
            "stop_reason": r["stop_reason"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "duration_seconds": r["duration_seconds"],
        }
        for i, (s, r, score) in enumerate(zip(samples, responses, scores))
    ]

    total_input = sum(r["input_tokens"] for r in responses)
    total_output = sum(r["output_tokens"] for r in responses)
    total_duration = sum(r["duration_seconds"] for r in responses)

    result = TaskResult(
        elapsed_seconds=elapsed,
        metrics=Metrics(accuracy=accuracy, accuracy_stderr=stderr),
        num_samples=n,
        samples_hash=samples_hash,
        task=task.name,
        modality=modality,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        tokens_per_second=(total_input + total_output) / total_duration
        if total_duration
        else 0.0,
    )
    return result, request_logs
