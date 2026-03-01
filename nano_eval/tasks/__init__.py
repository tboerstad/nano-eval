"""Task registry: maps modality name to Task instance."""

from nano_eval.tasks.chartqa import chartqa
from nano_eval.tasks.gsm8k import gsm8k_cot_llama

TASKS = {"text": gsm8k_cot_llama, "vision": chartqa}
