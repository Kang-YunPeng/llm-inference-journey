# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（使用清华源加速）
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/pytorch-wheels/cu121 \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# 复制应用代码
COPY src/ ./src/

# 启动命令
CMD ["python3", "src/main.py"]