from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class Logger:
    """最小日志器：本地 JSON/JSONL + 可选 wandb 标量。

    记录范围：config、responses、训练指标（loss, grad_norm 等）。
    """

    def __init__(
        self,
        run_id: str,
        results_root: str | Path = "results",
        wandb_run: Any | None = None,
    ):
        """初始化日志器并创建运行目录。

        Args:
            run_id: 运行唯一标识。
            results_root: 本地结果根目录。
            wandb_run: 可选 wandb 运行实例。
        """
        self.run_dir = Path(results_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

    def log_config(self, config: Any) -> None:
        """保存 config 快照到 config.json。

        Args:
            config: 配置对象（dataclass）或字典。
        """
        payload = (
            asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log_responses(
        self, global_step: int, problem_ids: list[str], response_ids: list[list[int]]
    ) -> None:
        """保存 student responses 到 responses_{global_step}.jsonl。

        Args:
            global_step: 全局训练步数。
            problem_ids: 问题 ID 列表。
            response_ids: 模型生成的 token IDs。Shape: [B, response_len]
        """
        path = self.run_dir / f"responses_{global_step}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pid, rids in zip(problem_ids, response_ids):
                json.dump(
                    {"problem_id": pid, "response_ids": rids}, f, ensure_ascii=False
                )
                f.write("\n")

    def log_step(self, global_step: int, **metrics: float) -> None:
        """记录训练指标到对应的 .jsonl 文件及 wandb。

        Args:
            global_step: 全局训练步数。
            **metrics: 指标名-值对（如 loss=0.5, grad_norm=1.2）。
        """
        for name, value in metrics.items():
            with open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
                json.dump(
                    {"global_step": global_step, name: value}, f, ensure_ascii=False
                )
                f.write("\n")
        if self.wandb_run is not None:
            self.wandb_run.log(
                {**metrics, "global_step": global_step}, step=global_step
            )
