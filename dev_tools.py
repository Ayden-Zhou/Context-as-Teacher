"""
dev_tools.py
开发辅助工具集。集成版本控制、代码质量检查等 pipeline。

Usage:
    python src/dev_tools.py push
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire

# 配置常量
LOG_FILE = Path("commit_logs.md")
SRC_DIR = Path("src")


class DevTools:
    """开发工具链封装"""

    def _run_cmd(self, cmd: list[str], check: bool = True, error_msg: str = "") -> bool:
        """执行 Shell 命令的辅助函数。

        Args:
            cmd: 命令行参数列表。
            check: 是否检查返回码。
            error_msg: 错误提示信息。

        Returns:
            是否执行成功。
        """
        try:
            subprocess.run(cmd, check=check, text=True, capture_output=False)
            return True
        except subprocess.CalledProcessError as e:
            if error_msg:
                print(f"❌ {error_msg}")
                print(f"   Error details: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}. Please install required tools.")
            return False

    def _check_quality(self) -> bool:
        """[Gatekeeper] 运行 Ruff 格式化与逻辑检查。

        Returns:
            质量检查是否通过。
        """
        print("🔍 [Quality Gate] Running Ruff...")

        # 1. Format (自动修复 Import 排序和格式)
        print("   - Auto-formatting (isort/black rules)...")
        # 按照 code_style.md，Pipeline 执行 ruff format (Auto-fix imports/layout)
        if not self._run_cmd(
            ["ruff", "check", ".", "--select", "I", "--fix"], check=False
        ):
            return False
        if not self._run_cmd(["ruff", "format", "."], check=False):
            return False

        # 2. Lint (严格检查逻辑错误，如 F401, F821)
        print("   - Checking logic consistency...")
        # 只检查 src 目录，忽略测试文件可能存在的宽松写法
        # 按照 code_style.md，ZERO F errors (F821/F401)
        if not self._run_cmd(
            ["ruff", "check", str(SRC_DIR), "--select", "F"],
            error_msg="Quality Gate Failed",
        ):
            return False

        return True

    def commit(self):
        """[Pipeline] 自动提交。

        流程: 读取日志 -> 质量门控 -> 构造 Commit -> 清空日志
        """
        # 1. 检查日志
        if not LOG_FILE.exists():
            print("ℹ️ No commit_logs.md found. Creating empty one.")
            LOG_FILE.touch()
            return

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            raw_logs = f.readlines()

        logs = [line.strip() for line in raw_logs if line.strip()]

        if not logs:
            print("ℹ️ Commit logs are empty. Nothing to commit.")
            return

        print(f"📝 Found {len(logs)} pending changes.")

        # 2. 质量门控 (Ruff)
        if not self._check_quality():
            print("🛑 Commit Aborted: Code quality checks failed.")
            sys.exit(1)

        # 3. 构造 Commit Message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        first_log = logs[0]
        if ":" in first_log:
            clean_log = (
                first_log.split("]", 1)[-1].strip() if "]" in first_log else first_log
            )
            title = f"auto: {clean_log[:60]}"
        else:
            title = f"auto: Batch update {timestamp}"

        body = "\n".join(logs)
        commit_msg = f"{title}\n\n[Auto-Commit Logs]\n{body}"

        # 4. Git 操作
        print("🚀 Executing Git commit...")

        if not self._run_cmd(["git", "add", "."], error_msg="Git add failed"):
            sys.exit(1)

        if self._run_cmd(["git", "commit", "-m", commit_msg], check=False):
            print("✅ Successfully committed.")
            # 5. 成功后清空日志
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("")
            print("🧹 commit_logs.md cleared.")
        else:
            print("❌ Git commit failed.")
            sys.exit(1)

    def push(self):
        """[Pipeline] 自动推送。

        流程: Git Push
        """
        print("🚀 Executing Git push...")
        if self._run_cmd(["git", "push"], error_msg="Git push failed"):
            print("✅ Successfully pushed to remote.")
        else:
            sys.exit(1)


if __name__ == "__main__":
    # 使用 Fire 暴露命令行接口
    fire.Fire(DevTools)
