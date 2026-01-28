
# 项目规范：上下文感知自蒸馏 (Context-Aware Self-Distillation)

**SYSTEM NOTE**: This document is optimized for AI Coding Agents. Content must be strictly technical, dense, and devoid of conversational filler or formatting flourishes. Maintain maximum information density.

## 1. 数学模型

* **类型**：Top-K Logit 蒸馏（近似逆向 KL 散度）。

* **架构**：单模型自蒸馏。$\pi_S$（学生）和 $\pi_T$（教师）共享权重 $\theta$。

* **输入**：

  * 学生：$x$（问题）

  * 教师：$m \oplus x$（记忆/系统提示词 + 问题）

* **目标函数**：在每个 token 步长 $t$ 上最小化 Top-K token 的逆向 KL 散度。

    $$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{t} D_{KL}^{(K)}(\pi_S(\cdot|x_{<t}) || \pi_T(\cdot|m, x_{<t})) \right]$$
    $$D_{KL}^{(K)}(P_S || P_T) = \sum_{v \in \text{Top-K}(P_S)} \tilde{P}_S(v) \cdot (\log \tilde{P}_S(v) - \log \tilde{P}_T(v))$$

  * $\tilde{P}$ 表示在 Top-K 集合上重新归一化后的概率值。

  * *约束*：$P_T$ 为常数（梯度分离）。

  * *区别*：Top-K 近似在保留主导模式的同时减少了计算量。比全词表蒸馏更高效，比单 token 采样更准确。

## 2. 实现规范（核心）

### 基础框架

* **生成**：`vLLM`（高效采样，PagedAttention）。
* **训练**：`trl.GKDTrainer`（HuggingFace，支持梯度）。
* **模型**：单个 `AutoModelForCausalLM`，vLLM 和 HF 共享权重（定期同步）。
* **配置**：`GKDConfig(beta=0.0)`（纯策略内 KL）。

### 整体逻辑（`src/main.py`）

```
┌───────────────────────────────────────────────────────────────────┐
│  Outer Loop (Memory Evolution)                                    │
│  for epoch in range(num_epochs):                                  │
│    ┌───────────────────────────────────────────────────────────┐  │
│    │  Inner Loop (Distillation Training)                       │  │
│    │  for batch in dataloader:                                 │  │
│    │    sample.generate(batch) → batch.response_ids            │  │
│    │    HF Student(batch.input_ids) → student_log_probs        │  │
│    │    HF Teacher([M] + input_ids) → teacher_log_probs        │  │
│    │    loss = TopK_ReverseKL(S, T) → backward → update θ_HF   │  │
│    │    if step % sync_interval == 0: sync(θ_HF → θ_vLLM)      │  │
│    └───────────────────────────────────────────────────────────┘  │
│    ┌───────────────────────────────────────────────────────────┐  │
│    │  Memory Update                                            │  │
│    │  TODO                                                     │  │
│    └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

**数据流** (以 `Batch` 为载体)：

``` text
DataLoader
    │
    ▼ dict[str, list[str]]
Batch(questions, answers, solutions)
    │
    ▼ prompt.build_prompts(batch)
batch.prompts: list[str]
    │
    ▼ tokenizer(batch.prompts)
batch.prompt_ids: Tensor[B, prompt_len]
    │
    ▼ sample.generate(batch)  # vLLM, detokenize=False
batch.response_ids: Tensor[B, response_len]
    │
    ▼ torch.cat([prompt_ids, response_ids], dim=1)
batch.input_ids: Tensor[B, seq_len]
    │
    ├──► model(input_ids) → logits[:, prompt_len-1:-1, :] → student_log_probs [grad=T]
    │
    └──► model([memory_ids, input_ids]) → logits[:, mem_len+prompt_len-1:-1, :] → teacher_log_probs [grad=F]
                │
                ▼
           TopK_ReverseKL(S, T) → loss → backward
```

**权重同步**：每 `sync_interval` 步 `θ_HF → θ_vLLM`。

### `sample.generate(batch: Batch) -> Batch`

```python
SamplingParams(temperature=1.0, max_tokens=512, detokenize=False)
# 填充 batch.response_ids: Tensor[B, response_len]
```

### `trainer.compute_loss(model, batch: Batch) -> Tensor`

``` text
前置: batch.is_ready_for_train() == True

Student: model(batch.input_ids)
         logits[:, prompt_len-1:-1, :] → log_softmax → student_log_probs
         grad=True

Teacher: model(cat[memory_ids, input_ids])
         logits[:, mem_len+prompt_len-1:-1, :] → log_softmax → teacher_log_probs
         grad=False (torch.no_grad)

Loss:    topk_vals, topk_idx = student_log_probs.topk(K)
         teacher_topk = teacher_log_probs.gather(-1, topk_idx)
         kl = (softmax(topk_vals) * (log_softmax(topk_vals) - log_softmax(teacher_topk))).sum(-1)
         return kl.mean()
```

### `sync_weights(hf_model, vllm_engine)`

每 N 步或每 epoch 将 HF 权重复制到 vLLM。

## 3. 目录结构

``` text
src/
├── main.py                      # 入口，训练循环
└── context_as_teacher/
    ├── dataclass.py             # Batch 数据类
    ├── trainer.py               # compute_loss: HF 双前向 + Top-K KL
    ├── sample.py                # generate: vLLM 采样 → batch.response_ids
    ├── sync.py                  # sync_weights: θ_HF → θ_vLLM
    ├── memory.py                # CachedMemory: prompt 存储 + 版本
    └── prompt.py                # build_prompts: → batch.prompts
```

## 4. 依赖项

* `torch`
* `transformers`
* `trl`
* `vllm`（采样引擎）
* `deepspeed`（推荐 ZeRO-2）
* `accelerate`
