nano-eval is for checking that a model behind an OpenAI chat/complations API is correctly implemented. 

```bash
nano-eval --tasks gsm8k_cot_llama --base_url=http://localhost:8000/v1 --max_samples 100

# prints:
{
  "results": {
    "gsm8k_cot_llama": {
      "metrics": {"exact_match": 0.85},
      "num_samples": 100,
      "elapsed": 45.2
    }
  },
  "config": {
    "model": "gpt-4", # Autodetected if only a single model is available
    "max_samples": 100
  },
  "total_seconds": 45.2
}

```

## Supported Tasks

| Task | Type | Dataset | Description |
|------|------|---------|-------------|
| `gsm8k_cot_llama` | Text | gsm8k | Grade school math with chain-of-thought (8-shot) |
| `chartqa` | Multimodal | HuggingFaceM4/ChartQA | Chart question answering with images |

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# Text and Image evals, with custom parameters
nano-eval \
    --tasks gsm8k_cot_llama,chartqa \
    --num_concurrent 64 \ # 64 max concurrent requests 
    --model_args model=llama-3,base_url=http://localhost:8000/v1 \
    --gen_kwargs temperature=0.7,max_tokens=1024 # Custom args passed through \
    --output results.json
```

### Python API

```python
import asyncio
from core import APIConfig, run_task
from tasks import TASKS

BASE_URL="http://localhost:8000/v1"

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
| `--model_args` | Config: `model="...",base_url="...",api_key="...",num_concurrent=8` | Required |
| `--gen_kwargs` | Generation params: `temperature=0.7,max_tokens=1024` | "" |
| `--max_samples` | Limit number of samples | None |
| `--seed` | Random seed | 42 |
| `--output` | Output JSON file | None |


This tool is inspired and borrows from: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

> **Note:** This tool is designed for **comparing relative accuracy between inference frameworks** (e.g., vLLM vs SGLang vs MAX running the same model). It is not intended for absolute benchmark evaluations or leaderboard submissions. Use it to verify that different serving backends produce consistent results.

