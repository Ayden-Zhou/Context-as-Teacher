"""
main.py
项目入口：RL 风格的 Rollout + Train 循环。

流程：rollout(batch_size * num_responses) → train(gradient_steps) → 循环
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from context_as_teacher.dataclass import Batch
from context_as_teacher.prompt import build_prompt_ids, get_tokenizer
from context_as_teacher.sample import generate_rollout
from context_as_teacher.trainer import ContextDistillationTrainer
from logger import Logger
from utils import (
    Timer,
    create_dataloader,
    generate_run_id,
    infinite_dataloader,
    set_global_seed,
)

# ==================== 配置 ====================


@dataclass
class Config:
    """训练配置"""

    # 模型
    model_path: str = "models/Qwen2.5-0.5B-Instruct"
    checkpoint_dir: str = "models/checkpoints"
    save_rollout_freq: int = 2  # 每 N 轮 rollout 保存一次model快照
    # 数据
    data_path: str = "data/dataset/gsm8k_train.jsonl"
    # RL 风格训练参数
    batch_size: int = 256  # 每个 gradient step 的 batch 大小
    responses_per_prompt: int = (
        16  # 每题采样响应数，总生成 batch_size * num_responses 条
    )
    gradient_steps: int = 16  # 每次 rollout 后训练的步数
    num_rollouts: int = 100  # 总 rollout 轮数
    # 生成参数
    max_new_tokens: int = 5120
    temperature: float = 1.0
    # 蒸馏参数
    top_k: int = 50
    learning_rate: float = 1e-5
    # 其他
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    pad_token_id: int = 0
    level: int = 0

    @property
    def rollout_size(self) -> int:
        """每次 rollout 生成的总样本数"""
        return self.batch_size * self.responses_per_prompt


def main(cfg: Config):
    """Rollout → Train → 循环"""
    set_global_seed(cfg.seed)
    tokenizer = get_tokenizer(cfg.model_path)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    cfg.pad_token_id = pad_id

    model_name = Path(cfg.model_path).name
    dataset_name = f"{Path(cfg.data_path).stem}_{cfg.level}"
    run_id = generate_run_id(model_name, dataset_name)

    run_root = Path(cfg.checkpoint_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = run_root / "latest"

    load_checkpoint = Path(cfg.model_path)
    print(f"[init] Run ID: {run_id}")

    # 初始化 Logger
    logger = Logger(run_id=run_id, results_root="results")
    logger.log_config(cfg)

    with Timer("load_data", start="[init] 加载数据集..."):
        dataloader = create_dataloader(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        data_iter = infinite_dataloader(dataloader)

    # memory = CachedMemory()
    trainer = ContextDistillationTrainer(
        cfg=cfg,
        model_root=run_root,
        latest_checkpoint=latest_checkpoint,
        load_checkpoint=load_checkpoint,
    )
    global_step = 0
    rollout_count = 0

    # ===== 主循环 =====
    while rollout_count < cfg.num_rollouts:
        # Phase 1 Prepare Prompts
        with Timer(
            label="prepare_prompts",
            accumulate=True,
            start=f"[step {global_step}] Prepare Prompts...",
        ):
            batch: Batch = next(data_iter)
            batch.student_prompt_ids, batch.teacher_prompt_ids = build_prompt_ids(
                problems=batch.problems,
                memories=batch.memories,
                model_path=trainer.load_checkpoint,
                responses_per_prompt=cfg.responses_per_prompt,
            )
            batch = batch.repeat_interleave(cfg.responses_per_prompt)
        # Phase 2: Rollout (vLLM)
        with Timer(
            label="rollout", accumulate=True, start=f"[step {global_step}] Rollout..."
        ):
            response_ids = generate_rollout(
                prompt_ids=batch.student_prompt_ids,
                checkpoint=trainer.load_checkpoint,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                responses_per_prompt=cfg.responses_per_prompt,
            )
            rollout_buffer = batch
            rollout_buffer.response_ids = response_ids
            # 记录 responses
            logger.log_responses(
                global_step=global_step,
                problem_ids=rollout_buffer.problem_ids,
                response_ids=response_ids,
            )
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 3: Train
        with Timer(
            label="train",
            accumulate=True,
            start=f"[step {global_step}] Train {cfg.gradient_steps} steps...",
        ):
            trainer.start_train()
            for mini_batch in rollout_buffer.sample_batches(
                cfg.batch_size, cfg.gradient_steps, cfg.device
            ):
                global_step, loss_val, grad_norm = trainer.update(
                    global_step,
                    prompt_ids=mini_batch.student_prompt_ids,
                    response_ids=mini_batch.response_ids,
                    teacher_prompt_ids=mini_batch.teacher_prompt_ids,
                )
                logger.log_step(global_step, loss=loss_val, grad_norm=grad_norm)
            rollout_count += 1
            trainer.finish_train(global_step, rollout_count)

    # ===== 保存最终模型 =====
    with Timer(label="save", accumulate=True, start="[done] 保存最终模型..."):
        output_dir = Path("outputs") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # 复制 checkpoint 到 outputs
        import shutil

        final_source = (
            latest_checkpoint
            if (latest_checkpoint / "config.json").exists()
            else load_checkpoint
        )
        shutil.copytree(final_source, output_dir, dirs_exist_ok=True)
        print(f"       模型已保存至 {output_dir}")

    # 耗时统计
    print("\n===== 耗时统计 =====")
    for label, secs in Timer.registry.items():
        print(f"  {label}: {secs:.2f}s")


if __name__ == "__main__":
    from fire import Fire

    Fire(lambda **kw: main(Config(**kw)))
