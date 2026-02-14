"""Task registry: maps modality name to Task instance."""

from tasks.chartqa import chartqa
from tasks.gsm8k import gsm8k_cot_llama

__all__: list[str] = []

TASKS = {"text": gsm8k_cot_llama, "vision": chartqa}
