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

```
$ nano-eval --help
Usage: nano-eval [OPTIONS]

  Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

  Example: nano-eval -t gsm8k_cot_llama --base-url http://localhost:8000/v1

  Use --version to display the version and exit.

Options:
  --version                       Show the version and exit.
  -t, --tasks [gsm8k_cot_llama|chartqa]
                                  Task to evaluate (can be repeated)
                                  [required]
  --base-url TEXT                 OpenAI-compatible API endpoint  [required]
  --model TEXT                    Model name; auto-detected if endpoint serves
                                  one model
  --api-key TEXT                  Bearer token for API authentication
  --num-concurrent INTEGER        Parallel requests to send  [default: 8]
  --max-retries INTEGER           Retry attempts for failed requests
                                  [default: 3]
  --extra-request-params TEXT     API params as key=value,...  [default:
                                  temperature=0,max_tokens=256,seed=42]
  --max-samples INTEGER           If provided, limit samples per task
  --output-path PATH              Write results.json and sample logs to this
                                  directory
  --log-samples                   Save per-sample results as JSONL (requires
                                  --output-path)
  --seed INTEGER                  Seed for shuffling samples  [default: 42]
  --help                          Show this message and exit.
```

### Python API

```python
import asyncio
from nano_eval import evaluate, EvalResult

result: EvalResult = asyncio.run(evaluate(
    tasks=["gsm8k_cot_llama"],
    base_url="http://localhost:8000/v1",
    model="gpt-4",
    max_samples=100,
))
gsm8k = result["results"]["gsm8k_cot_llama"]
print(f"Accuracy: {gsm8k['metrics']['exact_match']:.1%}")
```


This tool is inspired and borrows from: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Please check it out


