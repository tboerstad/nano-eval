"""
Task registry for nano-eval.

Maps type to Task instances:
- text: gsm8k_cot_llama (8-shot grade school math with chain-of-thought)
- vision: chartqa (multimodal chart understanding)
"""

from tasks.chartqa import chartqa
from tasks.gsm8k import gsm8k_cot_llama

TASKS = {"text": gsm8k_cot_llama, "vision": chartqa}

__all__ = ["TASKS", "gsm8k_cot_llama", "chartqa"]
