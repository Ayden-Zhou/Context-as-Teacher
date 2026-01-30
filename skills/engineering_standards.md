# Engineering Standards & Tech Stack (2026 Research Ed.)

**SYSTEM NOTE**: This document is optimized for AI Coding Agents. Content must be strictly technical, dense, and devoid of conversational filler or formatting flourishes. Maintain maximum information density.
**TARGET**: NVIDIA H800 (Hopper).
**PRIORITY**: Algorithm Correctness > Throughput.

## 0. Hardware Constraints

* **Arch**: Compute Capability 9.0 (Hopper).
* **VRAM**: 80GB HBM3.
* **Attention**: Native FA3 capable. Requires `flash-attn>=2.6.0`. Fallback: FA2.

## 1. Lifecycle & Reproducibility

* **Seeds**: Set global seeds (Py/Np/Torch) at entry.
* **Resource Protocol (Single-GPU)**: `del vllm_engine` -> `gc.collect()` -> `torch.cuda.empty_cache()` before Trainer init.

## 2. Inference (vLLM)

* **Dtype**: `bfloat16`.
* **Input**: `student_prompt_ids` (List[List[int]]).
* **KV Cache**: `auto` (No FP8/Quant).
* **Mem**: `gpu_memory_utilization=0.90`.

## 3. Training (PyTorch Native)

* **TF32**: `torch.set_float32_matmul_precision('high')`.
* **AMP**: `autocast('cuda', dtype=torch.bfloat16)`. **GradScaler**: DISABLED.
* **Optimizer**: `AdamW(fused=True)`. IF runtime_error THEN fallback `AdamW(fused=False)`.
* **Attention**: `attn_implementation="flash_attention_2"`. IF unsupported THEN `sdpa`.
* **GC**: `gradient_checkpointing_enable()` IF model > 4B.

## 4. Model & Data (HF Transformers)

* **Loader**: `trust_remote_code=True`.
* **Padding (Training)**: **RIGHT-Padding**. Mask pad tokens in Loss.
* **Tokenization**: CPU Pre-tokenize -> GPU ID Batch.
* **Save**: `safe_serialization=True`.