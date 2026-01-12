**nano-eval** is a minimal framework for evaluating text or vision models via OpenAI-compatible APIs.

## Quickstart

```python
import asyncio
from core import Task, Sample, TextPrompt
from tasks import TASKS
from nano_eval import evaluate

# Define a custom task
def my_samples(max_samples=None, seed=None):
    return [
        Sample(prompt=TextPrompt(text="What is 2+2?"), target="4"),
        Sample(prompt=TextPrompt(text="What is 3+3?"), target="6"),
    ]

def my_score(response: str, target: str) -> float:
    return 1.0 if target in response else 0.0

# Register the task
TASKS["math"] = Task(
    name="simple_math",
    task_type="text",
    samples=my_samples,
    score=my_score,
)

# Run evaluation
result = asyncio.run(evaluate(
    types=["math"],
    base_url="http://localhost:8000/v1",
    model="your-model",
))
print(f"Accuracy: {result['results']['math']['metrics']['exact_match']:.1%}")
```

## Usage

```
$ nano-eval --help
Usage: nano-eval [OPTIONS]

  Evaluate LLMs on custom tasks via OpenAI-compatible APIs.

  Example: nano-eval -t mytask --base-url http://localhost:8000/v1

Options:
  -t, --type TEXT               Task type to evaluate (can be repeated). Must
                                be registered in TASKS.  [required]
  --base-url TEXT               OpenAI-compatible API endpoint  [required]
  --model TEXT                  Model name; auto-detected if endpoint serves
                                one model
  --api-key TEXT                Bearer token for API authentication
  --max-concurrent INTEGER      [default: 8]
  --extra-request-params TEXT   API params as key=value,...  [default:
                                temperature=0,max_tokens=256,seed=42]
  --max-samples INTEGER         If provided, limit samples per task
  --output-path PATH            Write results.json and sample logs to this
                                directory
  --log-samples                 Save per-sample results as JSONL (requires
                                --output-path)
  --seed INTEGER                Controls sample order  [default: 42]
  -v, --verbose                 Increase verbosity (up to -vv)
  --version                     Show the version and exit.
  --help                        Show this message and exit.
```

## Core Concepts

### Task

A task defines what to evaluate:

```python
from core import Task, Sample, TextPrompt, VisionPrompt

Task(
    name="my_task",           # Unique identifier
    task_type="text",         # "text" or "vision"
    samples=my_samples_fn,    # (max_samples, seed) -> list[Sample]
    score=my_score_fn,        # (response, target) -> float
)
```

### Sample

A sample is a single evaluation item:

```python
# Text sample
Sample(prompt=TextPrompt(text="What is 2+2?"), target="4")

# Vision sample (with PIL images)
Sample(prompt=VisionPrompt(text="Describe this image", images=[pil_image]), target="A cat")
```

## Example Output

When using `--output-path`, a `results.json` file is generated:

```json
{
  "config": {
    "max_samples": 100,
    "model": "your-model"
  },
  "framework_version": "0.2.4",
  "results": {
    "math": {
      "elapsed_seconds": 5.23,
      "metrics": {
        "exact_match": 0.85,
        "exact_match_stderr": 0.036
      },
      "num_samples": 100,
      "samples_hash": "abc123...",
      "task": "simple_math",
      "task_type": "text"
    }
  },
  "total_seconds": 5.23
}
```

This tool is inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
