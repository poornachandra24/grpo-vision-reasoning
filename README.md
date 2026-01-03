# GRPO Vision Reasoning

## Setup & Build

We use a **Two-Stage Build Strategy**.
1.  **Stage 1:** Build the official vLLM image from source to handle ROCm dependencies correctly.
2.  **Stage 2:** Build our lightweight project layer on top.

### 1. Build Base vLLM Image (`vllm-mi300x-base`)

Clone the official vLLM repository and build the base image. Run this from a temporary directory (e.g., `~/workspace`):

```bash
# Clone vLLM
git clone https://github.com/vllm-project/vllm.git vllm-official
cd vllm-official

# Build Base Image (Takes ~15-20 mins)
DOCKER_BUILDKIT=1 \
docker build \
  -f docker/Dockerfile.rocm \
  --build-arg ARG_PYTORCH_ROCM_ARCH="gfx942" \
  --build-arg MAX_JOBS=16 \
  -t vllm-mi300x-base .
```

### 2. Build Project Image (`mi300x-grpo-env`)

Return to this repository root and build the final environment (Takes ~10 seconds):

```bash
cd ~/workspace/grpo-vision-reasoning
docker build -t mi300x-grpo-env .
```

## Run with Docker

To run the container with full GPU access and host networking:

```bash
docker run -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --shm-size=16g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --env-file .env \
    -v $(pwd):/workspace \
    mi300x-grpo-env
```