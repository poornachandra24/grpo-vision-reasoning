FROM vllm-mi300x-base:v0.11.2

# 1. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Ensure we use the vLLM environment created in the base image
    PATH="/opt/conda/bin:$PATH" \
    # Prevent fragmentation on MI300X
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    # ROCm optimizations for MI300X (gfx942)
    HSA_OVERRIDE_GFX_VERSION=9.4.2 \
    PYTORCH_ROCM_ARCH=gfx942

# 2. Install uv (for speed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Install Project Dependencies
# Note: TRL pinned to 0.11.x for vLLM 0.11.2 compatibility
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

# 4. Verify installation
RUN python -c "import vllm; print(f'✓ vLLM: {vllm.__version__}')" && \
    python -c "import trl; print(f'✓ TRL: {trl.__version__}')" && \
    python -c "from trl import GRPOTrainer; print('✓ GRPO imports: SUCCESS')"

# 5. Setup Workspace
WORKDIR /workspace

# 6. Copy project files (optional - can also use volume mount)
# COPY configs/ /workspace/configs/
# COPY src/ /workspace/src/

CMD ["/bin/bash"]