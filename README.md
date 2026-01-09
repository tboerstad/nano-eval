**nano-eval** is a minimal tool for measuring the quality of a text or vision model.

## Quickstart

```bash
uvx nano-eval -t gsm8k_cot_llama -t chartqa --base-url http://localhost:8000/v1 --max-samples 100

# prints:
Task    Accuracy  Samples  Duration
------  --------  -------  --------
text      84.3%      100       45s
vision    71.8%      100       38s
```

> **Note:** This tool is for eyeballing the accuracy of a model. One example use case is comparing relative accuracy between inference frameworks (e.g., vLLM vs SGLang vs MAX running the same model).

## Supported Tasks

| Task | Type | Dataset | Description |
|------|------|---------|-------------|
| `gsm8k_cot_llama` | Text | gsm8k | Grade school math with chain-of-thought (8-shot) |
| `chartqa` | Multimodal | HuggingFaceM4/ChartQA | Chart question answering with images |

## Usage

```
$ nano-eval --help
Usage: nano-eval [OPTIONS]

  Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

  Example: nano-eval -t gsm8k_cot_llama --base-url http://localhost:8000/v1

Options:
  -t, --tasks [gsm8k_cot_llama|chartqa]
                                  Task to evaluate (can be repeated)
                                  [required]
  --base-url TEXT                 OpenAI-compatible API endpoint  [required]
  --model TEXT                    Model name; auto-detected if endpoint serves
                                  one model
  --api-key TEXT                  Bearer token for API authentication
  --max-concurrent INTEGER        [default: 8]
  --extra-request-params TEXT     API params as key=value,...  [default:
                                  temperature=0,max_tokens=256,seed=42]
  --max-samples INTEGER           If provided, limit samples per task
  --output-path PATH              Write results.json and sample logs to this
                                  directory
  --log-samples                   Save per-sample results as JSONL (requires
                                  --output-path)
  --seed INTEGER                  Seed for shuffling samples  [default: 42]
  --version                       Show the version and exit.
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


