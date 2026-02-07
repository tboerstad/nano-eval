**nano-eval** is a minimal tool for measuring the quality of a text or vision model.

## Quickstart

```bash
uvx nano-eval -m text -m vision --max-samples 100

# prints:
Task    Accuracy  Samples  Duration  Output Tokens  Per Req Tok/s
------  --------  -------  --------  -------------  -------------
text      86.0%      100       15s          11873           7658
vision    72.0%      100       37s           8714           1894
```

> **Note:** This tool is for eyeballing the accuracy of a model. One use case is comparing accuracy between inference frameworks (e.g., vLLM vs SGLang vs MAX running the same model).

## Supported Modalities

| Modality | Dataset | Description |
|----------|---------|-------------|
| `text` | gsm8k_cot_llama | Grade school math with chain-of-thought (8-shot) |
| `vision` | HuggingFaceM4/ChartQA | Chart question answering with images |

## Usage

```
$ nano-eval --help
Usage: nano-eval [OPTIONS]

  Evaluate LLMs on standardized tasks via OpenAI-compatible APIs.

  Example: nano-eval -m text

Options:
  -m, --modality [text|vision]     Modality to evaluate (can be repeated)
                                  [required]
  --base-url TEXT                 OpenAI-compatible API endpoint; tries
                                  127.0.0.1:8000/8080 if omitted
  --model TEXT                    Model name; auto-detected if endpoint serves
                                  one model
  --api-key TEXT                  Bearer token for API authentication
  --max-concurrent INTEGER        [default: 8]
  --extra-request-params TEXT     API params as key=value,...  [default:
                                  temperature=0,max_tokens=256,seed=42]
  --max-samples INTEGER           If provided, limit samples per task
  --output-path PATH              Write eval_results.json and request logs to
                                  this directory
  --log-requests                  Save per-request results as JSONL (requires
                                  --output-path)
  --seed INTEGER                  Controls sample order  [default: 42]
  -v, --verbose                   Increase verbosity (up to -vvv)
  --version                       Show the version and exit.
  --help                          Show this message and exit.
```

### Python API

```python
from nano_eval import evaluate

result = evaluate(
    modalities=["text"],
    base_url="http://127.0.0.1:8000/v1",
    model="meta-llama/Llama-3.2-1B-Instruct",
    max_samples=100,
)
print(f"Accuracy: {result['results']['text']['metrics']['exact_match']:.1%}")
```

## Example Output

When using `--output-path`, an `eval_results.json` file is generated:

```json
{
  "config": {
    "max_samples": 100,
    "model": "deepseek-chat"
  },
  "framework_version": "0.2.6",
  "results": {
    "text": {
      "elapsed_seconds": 15.51,
      "metrics": {
        "exact_match": 0.86,
        "exact_match_stderr": 0.03487350880197947
      },
      "num_samples": 100,
      "samples_hash": "12a1e9404db6afe810290a474d69cfebdaffefd0b56e48ac80e1fec0f286d659",
      "task": "gsm8k_cot_llama",
      "modality": "text",
      "total_input_tokens": 106965,
      "total_output_tokens": 11873,
      "tokens_per_second": 7658.994842036105
    }
  },
  "total_seconds": 15.51
}
```

With `--log-requests`, a `request_log_{modality}.jsonl` is written per modality:

```json
{
  "request_id": 0,
  "target": "4",
  "prompt": "What is 2+2?",
  "response": "4",
  "exact_match": 1.0,
  "stop_reason": "stop",
  "input_tokens": 7,
  "output_tokens": 1,
  "duration_seconds": 0.83
}
```

---

Inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
