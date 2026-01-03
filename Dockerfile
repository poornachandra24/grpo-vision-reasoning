FROM vllm-mi300x-base

# 1. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Ensure we use the vLLM environment created in the base image
    PATH="/opt/conda/bin:$PATH" 

# 2. Install uv (for speed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Install Project Dependencies
# We install directly into the environment vLLM created
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

# 4. Setup Workspace
WORKDIR /workspace
CMD ["/bin/bash"]