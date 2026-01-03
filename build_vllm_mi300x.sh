#!/bin/bash
# Stage 1: Build vLLM Base Image for MI300X
# Version: v0.11.2 (TRL-compatible + has ROCm Dockerfile)

set -e

echo "=========================================="
echo "Building vLLM Base Image for MI300X"
echo "Version: 0.11.2 (TRL-compatible)"
echo "=========================================="
echo ""

# 1. Clone vLLM to temporary location
echo "📥 Cloning vLLM repository..."
cd ~/workspace
rm -rf vllm-official  # Clean up if exists
git clone https://github.com/vllm-project/vllm.git vllm-official
cd vllm-official

# 2. Checkout v0.11.2 (TRL-compatible + has ROCm Dockerfile)
echo "🔖 Checking out v0.11.2..."
git checkout v0.11.2

# 3. Verify Dockerfile exists
if [ ! -f "docker/Dockerfile.rocm" ]; then
    echo "❌ ERROR: docker/Dockerfile.rocm not found!"
    echo "Available docker files:"
    ls -la docker/
    exit 1
fi

echo "✅ Found docker/Dockerfile.rocm"
echo ""

echo "⚙️  Building Docker image (this takes 15-20 minutes)..."
echo "   Target: vllm-mi300x-base:v0.11.2"
echo "   Architecture: gfx942 (MI300X)"
echo "   ROCm support: Enabled"
echo ""

# 4. Build the Base Image
DOCKER_BUILDKIT=1 \
docker build \
  -f docker/Dockerfile.rocm \
  --build-arg ARG_PYTORCH_ROCM_ARCH="gfx942" \
  --build-arg MAX_JOBS=16 \
  -t vllm-mi300x-base:v0.11.2 .

echo ""
echo "✅ vLLM base image built successfully!"
echo "   Image: vllm-mi300x-base:v0.11.2"
echo "   Version: vLLM 0.11.2"
echo "   TRL compatibility: ✅ Full support"
echo ""
echo "Next step: Build Stage 2 (your project image)"
echo "   cd ~/workspace/grpo-vision-reasoning"
echo "   docker build -t mi300x-grpo-env:v0.11.2 ."