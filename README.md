**nano-eval** is a minimal tool for measuring the quality of a text or vision model. This is done by measuring the accuracy on hard coded datasets with known answers.

> **Note:** This tool is designed for **comparing relative accuracy between inference frameworks** (e.g., vLLM vs SGLang vs MAX running the same model). It is not intended for absolute benchmark evaluations (there's only two datasets). Use it to verify that different serving backends produce consistent results or track results over time. 

nano-eval tests against an OpenAI compliant endpoint, specifically the `chat/completions` API.

```bash
nano-eval --tasks gsm8k_cot_llama --base-url http://localhost:8000/v1 --max-samples 100

# prints:
{
  "results": {
    "gsm8k_cot_llama": {
      "task": "gsm8k_cot_llama",
      "task_hash": "abc123...",
      "metrics": {"exact_match": 0.85, "exact_match_stderr": 0.036},
      "num_samples": 100,
      "elapsed_seconds": 45.2
    }
  },
  "eval_hash": "def456...",
  "total_seconds": 45.2,
  "config": {
    "model": "gpt-4",
    "max_samples": 100
  }
}

```

## Supported Tasks

| Task | Type | Dataset | Description |
|------|------|---------|-------------|
| `gsm8k_cot_llama` | Text | gsm8k | Grade school math with chain-of-thought (8-shot) |
| `chartqa` | Multimodal | HuggingFaceM4/ChartQA | Chart question answering with images |

## Installation

```bash
pip install nano-eval
```

## Usage

### Command Line

```bash
# Text and Image evals, with custom parameters passed alongside the request
nano-eval \
    -t gsm8k_cot_llama -t chartqa \
    --base-url http://localhost:8000/v1 \
    --model llama-3 \
    --num-concurrent 64 \
    --extra-request-params temperature=0.7,max_tokens=1024 \
    --output-path ./results
```

### Python API

```python
import asyncio
from nano_eval import APIConfig, run_task, TASKS

BASE_URL = "http://localhost:8000/v1"

# Configure API endpoint
config = APIConfig(
    url=f"{BASE_URL}/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

# Run GSM8K evaluation
result = asyncio.run(run_task(TASKS["gsm8k_cot_llama"], config, max_samples=100))
print(f"GSM8K: {result['metrics']}")
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-t, --tasks` | Task to evaluate (can be repeated) | Required |
| `--base-url` | OpenAI-compatible API endpoint | Required |
| `--model` | Model name; auto-detected if endpoint serves one model | - |
| `--api-key` | Bearer token for API authentication | - |
| `--num-concurrent` | Parallel requests to send | 8 |
| `--max-retries` | Retry attempts for failed requests | 3 |
| `--extra-request-params` | Extra API params as `key=value,...` | - |
| `--max-samples` | Limit samples per task | all |
| `--output-path` | Write results.json and sample logs to this directory | - |
| `--log-samples` | Save per-sample results as JSONL (requires --output-path) | - |
| `--seed` | Seed for shuffling samples | 42 |


This tool is inspired and borrows from: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Please check it out


