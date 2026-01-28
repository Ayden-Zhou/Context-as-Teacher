# 项目任务自动化

# 默认任务：显示帮助
default:
    @just --list

# 初始化开发环境
setup:
    uv sync
    uv run pre-commit install

# 格式化代码
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# 检查代码（不修改）
check:
    uv run ruff format --check .
    uv run ruff check .

# 将 .py 转换为 .ipynb
notebook:
    uv run jupytext --to ipynb *.py

# 清理缓存
clean:
    rm -rf __pycache__ .ruff_cache .ipynb_checkpoints
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
