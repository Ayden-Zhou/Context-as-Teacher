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

# 安装 Miniconda 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 3. 接受 conda 服务条款并创建 Python 3.12 环境
RUN conda config --set solver classic \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n cat python=3.12 -y

# 设置默认 Shell，确保后续命令在 conda 环境中执行
SHELL ["conda", "run", "-n", "cat", "/bin/bash", "-c"]

# ============================================
# 4. 核心安装：vLLM (自动处理 PyTorch 依赖)
# ============================================
# 升级 pip 和 uv
RUN pip install --upgrade pip uv

# 直接安装 vllm。
# 它会自动拉取 torch 2.5.1+cu124 (或类似兼容版本)
# 也会自动安装适用于 H800 的 flash-attention
RUN uv pip install vllm

# ============================================
# 5. 安装其他项目依赖
# ============================================
# 安装 Dynasor (配置代理)
RUN http_proxy=http://iscs1411:14111234@202.120.40.41:9081 \
    https_proxy=http://iscs1411:14111234@202.120.40.41:9081 \
    uv pip install git+https://github.com/hao-ai-lab/Dynasor.git

# 安装数据处理库和工具
# 移除 transformers 版本锁定，让 vLLM 决定版本
RUN uv pip install pandas pyarrow numpy fire datasets

# 6. 设置工作空间
WORKDIR /workspace/cat

# 7. 配置启动环境
RUN conda init bash && echo "conda activate cat" >> ~/.bashrc

# 8. 入口
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cat"]
CMD ["/bin/bash"]