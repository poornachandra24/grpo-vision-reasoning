# Dockerfile
FROM rocm/dev-ubuntu-22.04:6.3.2

# 1. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ROCM_ARCH="gfx942" \
    ROCM_HOME="/opt/rocm" \
    PATH="/opt/rocm/bin:$PATH" \
    # Define the Virtual Environment location for uv
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

# 2. Install uv (The Speedup)
# We copy the binary directly from the official image. It's cleaner and faster.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. System Dependencies (Minimal)
RUN apt-get update && apt-get install -y \
    python3-dev \
    git \
    ninja-build \
    cmake \
    libnuma-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Create Virtual Environment & Install PyTorch
# uv creates the venv and installs torch in one go.
# We explicitly point to the ROCm wheel index.
WORKDIR /app
RUN uv venv && \
    uv pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2 \
    --extra-index-url https://pypi.org/simple

# 5. Install Build Dependencies & Triton
RUN uv pip install cmake ninja packaging setuptools wheel triton

# 6. Build vLLM from Source (The Compiler Bottleneck)
# uv speeds up the dependency install step here, but compilation is still C++.
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    # Fast dependency install
    uv pip install -r requirements-rocm.txt && \
    export PYTORCH_ROCM_ARCH="gfx942" && \
    # We use the venv python to trigger the build
    python3 setup.py install

# 7. Install Project Dependencies
# This step will now be almost instant thanks to uv's caching and resolution.
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

# 8. Final Setup
WORKDIR /workspace
CMD ["/bin/bash"]
