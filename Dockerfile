FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安裝基本套件與 Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-distutils python3-opencv \
    ca-certificates git wget sudo ninja-build \
    build-essential ffmpeg libsm6 libxext6 curl \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 jupyter（可與 pip 一起安裝）
RUN pip install --no-cache-dir --upgrade jupyter

# 建立使用者 (可選)
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser
WORKDIR /home/appuser

# 更新 pip 並安裝 Python 套件（使用 --no-cache-dir）
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir \
    tensorboard cmake onnx \
    torch==1.10.0+cu111 torchvision==0.11.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    'git+https://github.com/facebookresearch/fvcore' \
    matplotlib

# 安裝 Detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo

# Fix fvcore 快取目錄
ENV FVCORE_CACHE="/tmp"

# 預設工作目錄
WORKDIR /workspace
