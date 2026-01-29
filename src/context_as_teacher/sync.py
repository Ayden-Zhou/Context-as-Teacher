"""
sync.py
权重同步模块：θ_HF → θ_vLLM。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def sync_weights(hf_model: PreTrainedModel, vllm_sampler: Any) -> None:
    """将 HuggingFace 模型权重同步到 vLLM 引擎。
    
    Args:
        hf_model: HuggingFace 模型
        vllm_sampler: vLLM LLM 实例
    """
    # TODO: 实现权重同步
    # vllm_sampler.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(...)
    pass
