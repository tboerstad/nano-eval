**nano-eval** is a minimal tool for measuring the quality of a text or vision model.

## Quickstart

```bash
uvx nano-eval -t text -t vision --base-url http://localhost:8000/v1 --max-samples 100

# prints:
Task    Accuracy  Samples  Duration
------  --------  -------  --------
text      84.3%      100       45s
vision    71.8%      100       38s
```

> **Note:** This tool is for eyeballing the accuracy of a model. One use case is comparing accuracy between inference frameworks (e.g., vLLM vs SGLang vs MAX running the same model).

## Supported Types

| Type | Dataset | Description |
|------|---------|-------------|
| `text` | gsm8k_cot_llama | Grade school math with chain-of-thought (8-shot) |
| `vision` | HuggingFaceM4/ChartQA | Chart question answering with images |

## Usage

```
$ nano-eval --help
Usage: nano-eval [OPTIONS]

  Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

  Example: nano-eval -t text --base-url http://localhost:8000/v1

Options:
  -t, --type [text|vision]        Type to evaluate (can be repeated)
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
  --seed INTEGER                  Controls sample order  [default: 42]
  -v, --verbose                   Increase verbosity (up to -vv)
  --version                       Show the version and exit.
  --help                          Show this message and exit.
```

### Python API

```python
import asyncio
from nano_eval import evaluate, EvalResult

result: EvalResult = asyncio.run(evaluate(
    types=["text"],
    base_url="http://localhost:8000/v1",
    model="google/gemma-3-4b-it",
    max_samples=100,
))
text_result = result["results"]["text"]
print(f"Accuracy: {text_result['metrics']['exact_match']:.1%}")
```


This tool is inspired and borrows from: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Please check it out

## Example Output

When using `--output-path`, a `results.json` file is generated:

```json
{
  "config": {
    "max_samples": 37,
    "model": "google/gemma-3-4b-it"
  },
  "framework_version": "0.2.1",
  "results": {
    "text": {
      "elapsed_seconds": 28.45,
      "metrics": {
        "exact_match": 0.7837837837837838,
        "exact_match_stderr": 0.06861056852129647
      },
      "num_samples": 37,
      "samples_hash": "12a1e9404db6afe810290a474d69cfebdaffefd0b56e48ac80e1fec0f286d659",
      "task": "gsm8k_cot_llama",
      "task_type": "text"
    }
  },
  "total_seconds": 28.45
}
```
