FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive
# 禁用源码编译，确保使用 pip 预编译包
ENV VLLM_USE_PRECOMPILED=1 
ENV PYTHONUNBUFFERED=1
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# 安装系统依赖
# 新增 procps (提供 ps 命令，方便查看进程)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    procps \
    tmux \
    build-essential \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 使用 uv 安装 Python 3.12 并创建虚拟环境
ENV UV_PYTHON_DOWNLOADS=automatic
RUN uv python install 3.12

# 设置工作空间
WORKDIR /workspace/cat

# 创建虚拟环境
ENV VIRTUAL_ENV=/workspace/cat/.venv
RUN uv venv $VIRTUAL_ENV --python 3.12
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ============================================
# 安装项目依赖 (通过 pyproject.toml + uv.lock)
# ============================================
# 先复制依赖定义文件（利用 Docker 层缓存）
COPY pyproject.toml uv.lock ./

# 根据 lock 文件安装依赖（不更新 lock，不安装 dev 依赖）
# vllm 会自动拉取 torch 2.5.1+cu124 和 flash-attention
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 7. 配置启动环境 (虚拟环境通过 PATH 自动激活)
RUN echo "source $VIRTUAL_ENV/bin/activate" >> ~/.bashrc

# 8. 入口
CMD ["/bin/bash"]