"""
Send one ChartQA request using lm-eval's pipeline through the logging server.

This replicates the exact request format that lm-eval would send,
including:
- The lm-eval ChartQA prompt template (with <image> placeholder)
- The image encoding with "detail": "auto"
- The generation kwargs (max_tokens=512, etc.)

Usage:
    python debug/send_lm_eval.py [--port 9999]
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO

import httpx

LM_EVAL_CHARTQA_PROMPT_TEMPLATE = """\
<image>{query}
Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> follows the following instructions:
- <answer> should should be a single phrase or number.
- <answer> should not paraphrase or reformat the text in the image.
- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
- If the question is a Yes/No question, <answer> should be Yes/No.
- If <answer> is a number, it should not contain any units.
- If <answer> is a percentage, it should include a % sign.
- If <answer> is an entity, it should include the full label from the graph.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""


def encode_image_lm_eval(img):
    """Encode image exactly as lm-eval does (PNG, base64, with detail: auto)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"},
    }


def build_lm_eval_messages(query, image):
    """Build messages exactly as lm-eval does for ChartQA."""
    # lm-eval renders the Jinja template
    text = LM_EVAL_CHARTQA_PROMPT_TEMPLATE.format(query=query)

    # lm-eval builds the content array: images first, then text
    image_content = encode_image_lm_eval(image)
    text_content = {"type": "text", "text": text}

    messages = [{"role": "user", "content": [image_content, text_content]}]
    return messages


def main():
    port = 9999
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    # Load the same ChartQA sample that nano-eval would use
    # We use the same dataset and seed for a fair comparison
    print("Loading 1 ChartQA sample from HuggingFace...")
    import datasets

    datasets.utils.logging.set_verbosity_error()
    ds = datasets.load_dataset(
        "HuggingFaceM4/ChartQA",
        split="test",
        revision="b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
    )
    ds = ds.shuffle(seed=42)
    doc = ds[0]

    query = doc["query"]
    image = doc["image"]
    label = doc["label"]
    target = label[0] if isinstance(label, list) else str(label)

    print(f"Query: {query}")
    print(f"Target: {target}")
    print(f"Image type: {type(image)}")

    # Build messages as lm-eval would
    messages = build_lm_eval_messages(query, image)

    # lm-eval's generation kwargs for ChartQA
    payload = {
        "model": "debug-model",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "stop": [],
        "seed": 1234,
    }

    print(f"\nSending to {url}...")
    resp = httpx.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-Framework": "lm-eval",
        },
        timeout=30,
    )
    print(f"Response status: {resp.status_code}")
    print(f"Response: {resp.json()['choices'][0]['message']['content']}")
    print("Done! Check the logging server output.")


if __name__ == "__main__":
    main()
