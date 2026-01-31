# 项目任务自动化

VENV := justfile_directory()

# 默认任务：显示帮助
default:
    @just --list

# 初始化开发环境
setup:
    VIRTUAL_ENV={{VENV}}/.venv uv sync
    VIRTUAL_ENV={{VENV}}/.venv uv run pre-commit install

# 格式化代码
fmt:
    VIRTUAL_ENV={{VENV}}/.venv uv run ruff format .
    VIRTUAL_ENV={{VENV}}/.venv uv run ruff check --fix .

# 检查代码（不修改）
check:
    VIRTUAL_ENV={{VENV}}/.venv uv run ruff format --check .
    VIRTUAL_ENV={{VENV}}/.venv uv run ruff check .

# 将 .py 转换为 .ipynb
notebook:
    VIRTUAL_ENV={{VENV}}/.venv uv run jupytext --to ipynb *.py

# 清理缓存
clean:
    rm -rf __pycache__ .ruff_cache .ipynb_checkpoints
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

push:
    VIRTUAL_ENV={{VENV}}/.venv uv run dev_tools.py commit
    VIRTUAL_ENV={{VENV}}/.venv uv run dev_tools.py push

commit:
    VIRTUAL_ENV={{VENV}}/.venv uv run dev_tools.py commit

# 运行训练，支持覆盖任意 Config 字段
run *args:
    VIRTUAL_ENV={{VENV}}/.venv uv run src/main.py {{args}}