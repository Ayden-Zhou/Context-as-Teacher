"""
main.py
项目入口：RL 风格的 Rollout + Train 循环。

流程：rollout(batch_size * num_responses) → train(gradient_steps) → 循环
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from context_as_teacher.dataclass import Batch
from context_as_teacher.memory import CachedMemory
from context_as_teacher.sample import generate_rollout
from context_as_teacher.trainer import ContextDistillationTrainer
from utils import Timer


# ==================== 配置 ====================


@dataclass
class Config:
    """训练配置"""

    # 模型
    model_path: str = "models/Qwen2.5-0.5B-Instruct"
    checkpoint_dir: str = "checkpoints"
    save_model_freq: int = 10
    # 数据
    data_path: str = "data/dataset/gsm8k_train.jsonl"
    # RL 风格训练参数
    batch_size: int = 256        # 每个 gradient step 的 batch 大小
    responses_per_prompt: int = 16      # 每题采样响应数，总生成 batch_size * num_responses 条
    gradient_steps: int = 16     # 每次 rollout 后训练的步数
    total_steps: int = 1000      # 总训练步数
    # 生成参数
    max_new_tokens: int = 5120
    temperature: float = 1.0
    # 蒸馏参数
    top_k: int = 50
    learning_rate: float = 1e-5
    # 其他
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def rollout_size(self) -> int:
        """每次 rollout 生成的总样本数"""
        return self.batch_size * self.responses_per_prompt


# ==================== 数据加载 ====================


def collate_fn(batch: list[dict]) -> Batch:
    """将 HuggingFace Dataset 的 batch 转换为 Batch 对象"""
    return Batch(
        questions=[d["problem"] for d in batch],
        answers=[d.get("answer", "") for d in batch],
        solutions=[d.get("solution", "") for d in batch],
    )


def create_dataloader(cfg: Config) -> DataLoader:
    """创建无限循环的 DataLoader"""
    ds = load_dataset("json", data_files=cfg.data_path, split="train")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )


def infinite_dataloader(dataloader: DataLoader):
    """无限循环的数据迭代器"""
    while True:
        yield from dataloader


# ==================== 主函数 ====================


def main(cfg: Config):
    """Rollout → Train → 循环"""

    model_id = Path(cfg.model_path).name
    model_root = Path(cfg.checkpoint_dir) / model_id
    model_root.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = model_root / "latest"

    load_checkpoint = Path(cfg.model_path)

    with Timer("load_data", start="[init] 加载数据集..."):
        dataloader = create_dataloader(cfg)
        data_iter = infinite_dataloader(dataloader)

    memory = CachedMemory()
    trainer = ContextDistillationTrainer(
        cfg=cfg,
        memory=memory,
        model_root=model_root,
        latest_checkpoint=latest_checkpoint,
        load_checkpoint=load_checkpoint,
    )
    global_step = 0

    # ===== 主循环 =====
    while global_step < cfg.total_steps:

        # Phase 1: Rollout (vLLM)
        with Timer("rollout", start=f"[step {global_step}] Rollout..."):
            batch: Batch = next(data_iter)
            rollout_buffer: Batch = generate_rollout(batch, trainer.load_checkpoint, cfg)
        # rollout_buffer: Batch，总样本数 = batch_size * num_responses

        # Phase 2: Train (HuggingFace)
        with Timer("train", start=f"[step {global_step}] Train {cfg.gradient_steps} steps..."):
            response_batches = list(
                rollout_buffer.split(cfg.batch_size, shuffle=True)
            )
            if not response_batches:
                continue

            response_iter = cycle(response_batches)
            trainer.start_train()
            for i in range(cfg.gradient_steps):
                global_step = trainer.update(next(response_iter), global_step)
            trainer.finish_train(global_step)

        # Phase 3: Memory Update (TODO)
        # memory.update(...)

    # ===== 保存最终模型 =====
    with Timer("save", start="[done] 保存最终模型..."):
        output_dir = Path("outputs") / model_id
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
    main(Config())
