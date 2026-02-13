"""
Send one ChartQA request using nano-eval's pipeline through the logging server.

Usage:
    python debug/send_nano_eval.py [--port 9999]
"""

from __future__ import annotations

import sys

import httpx

from core import VisionPrompt, _build_vision_message
from tasks.chartqa import samples


def main():
    port = 9999
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    # Load one sample
    print("Loading 1 ChartQA sample...")
    sample_list = samples(max_samples=1, seed=42)
    sample = sample_list[0]
    prompt = sample.prompt
    assert isinstance(prompt, VisionPrompt)

    print(f"Query text (first 200 chars): {prompt.text[:200]}...")
    print(f"Target: {sample.target}")
    print(f"Image type: {type(prompt.images[0])}")
    print(f"Number of images: {len(prompt.images)}")

    # Build the messages exactly as nano-eval does
    messages = _build_vision_message(prompt.text, prompt.images)

    payload = {
        "model": "debug-model",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "seed": 42,
    }

    # Send with custom header to identify framework
    print(f"\nSending to {url}...")
    resp = httpx.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-Framework": "nano-eval",
        },
        timeout=30,
    )
    print(f"Response status: {resp.status_code}")
    print(f"Response: {resp.json()['choices'][0]['message']['content']}")
    print("Done! Check the logging server output.")


if __name__ == "__main__":
    main()
