# nano-eval

Nano Eval - A minimal harness for evaluating LLMs via OpenAI-compatible APIs.

> **Note:** This tool is designed for **comparing relative accuracy between inference frameworks** (e.g., vLLM vs TGI vs Ollama running the same model). It is not intended for absolute benchmark evaluations or leaderboard submissions. Use it to verify that different serving backends produce consistent results.

## Features

- **API-only** - works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, etc.)
- **Async** - concurrent requests with configurable parallelism
- **Two built-in tasks** - GSM8K (text) and ChartQA (multimodal)
- **Minimal** - lean codebase, few dependencies

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# Evaluate GSM8K math benchmark
nano-eval \
    --tasks gsm8k_cot_llama \
    --model_args model=gpt-4,base_url=http://localhost:8000/v1 \
    --max_samples 100

# Evaluate ChartQA (multimodal)
nano-eval \
    --tasks chartqa \
    --model_args model=gpt-4-vision,base_url=http://localhost:8000/v1,num_concurrent=4

# Both tasks with generation kwargs
nano-eval \
    --tasks gsm8k_cot_llama,chartqa \
    --model_args model=llama-3,base_url=http://localhost:8000/v1 \
    --gen_kwargs temperature=0.7,max_tokens=1024 \
    --output results.json
```

### Python API

```python
import asyncio
from core import APIConfig, run_task
from tasks import TASKS

# Configure API endpoint
config = APIConfig(
    url="http://localhost:8000/v1/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

# Run GSM8K evaluation
result = asyncio.run(run_task(TASKS["gsm8k_cot_llama"], config, max_samples=100))
print(f"GSM8K: {result['metrics']}")

# Run ChartQA evaluation
result = asyncio.run(run_task(TASKS["chartqa"], config, max_samples=100))
print(f"ChartQA: {result['metrics']}")
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tasks` | Comma-separated task names (gsm8k_cot_llama, chartqa) | Required |
| `--model_args` | Config: `model="...",base_url="...",api_key="...",num_concurrent=8` | Required |
| `--gen_kwargs` | Generation params: `temperature=0.7,max_tokens=1024` | "" |
| `--max_samples` | Limit number of samples | None |
| `--seed` | Random seed | 42 |
| `--output` | Output JSON file | None |

## Supported Tasks

| Task | Type | Dataset | Description |
|------|------|---------|-------------|
| `gsm8k_cot_llama` | Text | gsm8k | Grade school math with chain-of-thought (8-shot) |
| `chartqa` | Multimodal | HuggingFaceM4/ChartQA | Chart question answering with images |

## Output Format

```json
{
  "results": {
    "gsm8k_cot_llama": {
      "metrics": {"exact_match": 0.85},
      "num_samples": 100,
      "elapsed": 45.2
    }
  },
  "config": {
    "model": "gpt-4",
    "max_samples": 100
  },
  "total_seconds": 45.2
}
```
