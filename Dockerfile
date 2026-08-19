# syntax=docker/dockerfile:1
# SAM3 FastAPI 推理服务镜像
# 基础镜像已包含 CANN + Python + Ascend310P 运行环境；宿主机提供驱动。
#
# 可通过 --build-arg 指定 GitHub 镜像地址，加速国内构建
#   TOKENIZERS_GIT_REPOSITORY: tokenizers-cpp 仓库地址
#   ABSEIL_GIT_REPOSITORY:     abseil-cpp 仓库地址

ARG CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11
FROM ${CANN_IMAGE} AS builder

ARG TOKENIZERS_GIT_REPOSITORY=https://github.com/mlc-ai/tokenizers-cpp.git
ARG ABSEIL_GIT_REPOSITORY=https://github.com/abseil/abseil-cpp.git
ENV TOKENIZERS_GIT_REPOSITORY=${TOKENIZERS_GIT_REPOSITORY} \
    ABSEIL_GIT_REPOSITORY=${ABSEIL_GIT_REPOSITORY}

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    OPENCV_INSTALL_DIR=/usr

# 安装编译工具、OpenCV 以及 Python 构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libopencv-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 Rust 工具链（tokenizers-cpp 构建需要 cargo/rustc）
# 使用国内镜像加速 rustup 与 crates.io 下载
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup \
    RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static

RUN curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/rust-static/rustup/rustup-init.sh | sh -s -- -y --default-toolchain stable && \
    mkdir -p $CARGO_HOME && \
    printf '[registry]\ndefault = "ustc"\n\n[registries.ustc]\nindex = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"\n' > $CARGO_HOME/config.toml

# 配置国内 PyPI 镜像并安装 pybind11（C++ 扩展构建依赖）
RUN pip3 config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple && \
    pip3 install --no-cache-dir pybind11==2.12.0

# 仅复制构建 C++ 库所需的文件，避免 service/、readme 等变更触发重新编译
# third_party/tokenizers-cpp 会在 CMake 中按需拉取，不需要放入镜像上下文
WORKDIR /app
COPY CMakeLists.txt /app/
COPY src /app/src
COPY third_party /app/third_party

# 不依赖 BuildKit 的 RUN --mount，兼容服务器上的 Docker 18.09 / Compose 1.22。
RUN mkdir -p /app/build && cd /app/build && \
    PYBIND11_DIR=$(python3 -m pybind11 --cmakedir) && \
    PYTHON_BIN_DIR=$(python3 -c "import sys, os; print(os.path.dirname(sys.executable))") && \
    export PATH="${PYTHON_BIN_DIR}:${PATH}" && \
    cmake -Dpybind11_DIR=$PYBIND11_DIR \
          -DPYTHON_EXECUTABLE=$(which python3) \
          -DPython_EXECUTABLE=$(which python3) \
          -DPYBIND11_FINDPYTHON=ON \
          -DTOKENIZERS_GIT_REPOSITORY=$TOKENIZERS_GIT_REPOSITORY \
          -DABSEIL_GIT_REPOSITORY=$ABSEIL_GIT_REPOSITORY \
          -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) ascendsam3_py && \
    cp /app/build/ascendsam3*.so /app/

# ------------------------------------------------------------------------------
# 运行阶段：只保留 Python 服务依赖和编译好的 .so，镜像更小
FROM ${CANN_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:\
/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:\
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/develop/lib64:\
${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 配置国内 PyPI 镜像并安装 Python 服务依赖
RUN pip3 config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
RUN pip3 install --no-cache-dir -r /app/service/requirements.txt
COPY service /app/service
COPY --from=builder /app/ascendsam3*.so /app/

# 暴露 FastAPI 服务端口
EXPOSE 8000

# 启动服务；模型目录通过 docker-compose 挂载到 /app/models
CMD ["python3", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
