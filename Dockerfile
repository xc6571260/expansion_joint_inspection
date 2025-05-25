FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安裝 Python 3.10 和開發工具、OpenCV 依賴
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean

# 設定 python/python3 symlink，安裝 pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install --upgrade pip

# 若你有 requirements.txt，請先準備好
COPY requirements.txt .
RUN pip install -r requirements.txt

# 安裝 CUDA 12.4 專用 PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
