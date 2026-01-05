nano-eval is for checking that a model behind an OpenAI chat/completions API is correctly implemented. 

```bash
nano-eval --tasks gsm8k_cot_llama --base_url=http://localhost:8000/v1 --max_samples 100

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
# Text and Image evals, with custom parameters
nano-eval \
    --tasks gsm8k_cot_llama,chartqa \
    --base_url http://localhost:8000/v1 \
    --model llama-3 \
    --num_concurrent 64 \
    --gen_kwargs temperature=0.7,max_tokens=1024 \
    --output_path ./results
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
| `--tasks` | Comma-separated task names (gsm8k_cot_llama, chartqa) | Required |
| `--base_url` | API base URL (e.g. http://localhost:8000/v1) | Required |
| `--model` | Model name (auto-detected if API serves only one) | None |
| `--api_key` | API authentication key | "" |
| `--num_concurrent` | Max concurrent requests | 8 |
| `--max_retries` | Max retries per request | 3 |
| `--gen_kwargs` | Generation params: `temperature=0.7,max_tokens=1024` | temperature=0,max_tokens=256,seed=42 |
| `--max_samples` | Limit samples per task | None |
| `--output_path` | Directory for results.json and sample files | None |
| `--log_samples` | Write per-sample JSONL files | false |


This tool is inspired and borrows from: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

> **Note:** This tool is designed for **comparing relative accuracy between inference frameworks** (e.g., vLLM vs SGLang vs MAX running the same model). It is not intended for absolute benchmark evaluations or leaderboard submissions. Use it to verify that different serving backends produce consistent results.

