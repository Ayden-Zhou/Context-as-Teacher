
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
* **训练**：HuggingFace `Trainer`（支持梯度）。
* **解耦策略**：时间复用。vLLM 和 HF 不同时占用 GPU，交替加载释放。
* **配置**：纯策略内 Top-K KL。

### 整体逻辑（`src/main.py`）

RL 风格 Prepare → Rollout → Train 循环。每次生成 `batch_size × responses_per_prompt` 条 response，训练 `gradient_steps` 步。

``` text
┌─────────────────────────────────────────────────────────────────────────┐
│  RL-Style Training Loop                                                 │
│  while global_step < total_steps:                                       │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │  Phase 1: Prepare Prompts                                       │  │
│    │  batch = next(data_iter)  # batch_size 个问题                   │  │
│    │  student_prompt_ids, teacher_prompt_ids = build_prompt_ids(...) │  │
│    │  batch = batch.repeat_interleave(responses_per_prompt)          │  │
│    └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │  Phase 2: Rollout (vLLM)                                        │  │
│    │  response_ids = generate_rollout(student_prompt_ids, ...)       │  │
│    │  rollout_buffer.response_ids = response_ids                     │  │
│    │  # rollout_buffer: batch_size × responses_per_prompt 条 response │  │
│    └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │  Phase 3: Train (HuggingFace)                                   │  │
│    │  trainer.start_train()                                          │  │
│    │  for mini_batch in rollout_buffer.sample_batches(...):          │  │
│    │      global_step = trainer.update(global_step, prompt_ids,      │  │
│    │                        response_ids, teacher_prompt_ids)        │  │
│    │  trainer.finish_train(global_step)                              │  │
│    └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**参数配置**：  

* `batch_size = 256`：每个 gradient step 的样本数
* `responses_per_prompt = 16`：每题采样响应数，总生成 `256 × 16 = 4096` 条
* `gradient_steps = 16`：每次 rollout 后训练的步数
* `total_steps = 1000`：总训练步数
* `max_new_tokens = 5120`：生成最大 token 数
* `temperature = 1.0`：采样温度
* `top_k = 50`：蒸馏时 Top-K 值
* `learning_rate = 1e-5`：学习率

**显存策略**：单 GPU 时间复用。Rollout 阶段仅 vLLM 占用显存，Train 阶段仅 HF 占用显存。通过 checkpoint 目录共享权重。

**数据流** (以 `Batch` 为载体)：

``` text
Phase 1: Prepare Prompts
──────────────────────────────────────────────────────────────────
batch = next(data_iter)  # 取 batch_size 个问题
    │
    ▼ build_prompt_ids(problems, memories, model_path, responses_per_prompt)
batch.student_prompt_ids, batch.teacher_prompt_ids  # 预构建 prompt token ids
    │
    ▼ batch.repeat_interleave(responses_per_prompt)
batch  # 扩展为 batch_size * responses_per_prompt 条
──────────────────────────────────────────────────────────────────

Phase 2: Rollout (vLLM in GPU)
──────────────────────────────────────────────────────────────────
    │
    ▼ generate_rollout(prompt_ids, checkpoint, max_new_tokens, temperature, ...)
response_ids: list[list[int]]  # 生成的 token ids
    │
rollout_buffer = batch
rollout_buffer.response_ids = response_ids
──────────────────────────────────────────────────────────────────

Phase 3: Train (HF in GPU)
──────────────────────────────────────────────────────────────────
trainer.start_train()
for mini_batch in rollout_buffer.sample_batches(batch_size, gradient_steps, device):
    │
    ▼ trainer.update(global_step, prompt_ids, response_ids, teacher_prompt_ids)
    │
    ├──► model(student_prompt_ids + response_ids) → student_logits [grad=T]
    └──► model(teacher_prompt_ids + response_ids) → teacher_logits [grad=F]
                │
                ▼
           topk_reverse_kl(S, T) → loss → backward → step
──────────────────────────────────────────────────────────────────
trainer.finish_train(global_step)  # 保存 checkpoint
```

**权重共享**：通过 `checkpoint_dir` 目录。每次 Train 结束后保存至 `latest`，下次 Rollout 时加载。


### `generate_rollout(prompt_ids, checkpoint, ...) -> list[list[int]]`

```python
# main.py 调用方式
response_ids = generate_rollout(
    prompt_ids=batch.student_prompt_ids,
    checkpoint=trainer.load_checkpoint,
    max_new_tokens=cfg.max_new_tokens,
    temperature=cfg.temperature,
    responses_per_prompt=cfg.responses_per_prompt,
)
```

### `trainer.update(global_step, prompt_ids, response_ids, teacher_prompt_ids) -> int`

```python
# main.py 调用方式
trainer.start_train()
for mini_batch in rollout_buffer.sample_batches(cfg.batch_size, cfg.gradient_steps, cfg.device):
    global_step = trainer.update(
        global_step,
        prompt_ids=mini_batch.student_prompt_ids,
        response_ids=mini_batch.response_ids,
        teacher_prompt_ids=mini_batch.teacher_prompt_ids,
    )
trainer.finish_train(global_step)
```

## 3. 目录结构

``` text
src/
├── main.py                      # 入口：RL 风格 rollout + train 循环
└── context_as_teacher/
    ├── dataclass.py             # Batch: 数据载体，支持切片/堆叠/设备迁移
    ├── sample.py                # generate_rollout: vLLM 批量采样
    ├── trainer.py               # train_step: 单步蒸馏训练
    ├── memory.py                # CachedMemory: prompt 存储 + 版本
    └── prompt.py                # build_prompts: questions → prompts
```

## 4. 依赖项

* `torch`
* `transformers`
* `vllm`（采样引擎）
* `datasets`（数据加载）
* `accelerate`（可选，多卡训练）
