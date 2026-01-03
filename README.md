# 🦅 VLM-RL on MI300X: GRPO with Qwen2.5-VL

> **High-Performance Reinforcement Learning Pipeline for Vision-Language Models**  
> Optimized for AMD MI300X Accelerators | Powered by vLLM & TRL

This repository implements **Group Relative Policy Optimization (GRPO)** for fine-tuning **Qwen2.5-VL** using reinforcement learning. It is specifically engineered for the **AMD ROCm** ecosystem, leveraging `vLLM` for accelerated rollout generation and LoRA for memory-efficient training.

## ⚡ Key Features

*   **AMD MI300X Native**: Built on top of the official `rocm/vllm` stack for maximum performance on CDNA3 architecture.
*   **GRPO Implementation**: Uses TRL's `GRPOTrainer` to optimize reasoning policies without a value function, saving massive amounts of VRAM.
*   **vLLM Acceleration**: Integrated `vLLM` backend for rollout generation, providing ~10x speedup over standard Hugging Face generation.
*   **Inference CLI**: Powerful inference script with support for batching, registry lookup, WandB logging, and local result persistence.
*   **Parallel Data Prep**: Multi-process data loading to minimize startup time.
*   **Structured Reasoning**: Includes reward functions specifically designed to enforce `<reasoning>` and `<answer>` XML formats, similar to R1-style chain-of-thought.
*   **MLOps Ready**: Full integration with **Weights & Biases** for tracking and **Hugging Face Hub** for artifact versioning. Gradient checkpointing enabled by default.

---

## 📂 Project Structure

```text
├── configs/
│   └── qwen_grpo.yaml       # Experiment configuration (Hyperparams, LoRA, Checkpoints)
├── src/
│   ├── train.py             # Main entry point: Trainer setup, MLOps, & Execution
│   ├── model.py             # Model loading with QLoRA/LoRA adapters
│   ├── data.py              # Dataset formatting (MathVista/VLM datasets)
│   └── rewards.py           # Custom reward functions (Format, XML count, Correctness)
├── Dockerfile               # Production environment definition (Stage 2)
└── requirements.txt         # Python dependencies
```

---

## 🛠️ Installation & Build

We use a **Two-Stage Build Strategy** to ensure stability with ROCm dependencies.

### Stage 1: Build Base vLLM Image
First, we build the official vLLM image from source. This handles the complex compilation of PagedAttention for ROCm.

```bash
# 1. Clone vLLM to a temporary location
cd ~/workspace
git clone https://github.com/vllm-project/vllm.git vllm-official
cd vllm-official

# 2. Build the Base Image (Takes ~15-20 mins)
DOCKER_BUILDKIT=1 \
docker build \
  -f docker/Dockerfile.rocm \
  --build-arg ARG_PYTORCH_ROCM_ARCH="gfx942" \
  --build-arg MAX_JOBS=16 \
  -t vllm-mi300x-base .
```

### Stage 2: Build Project Image
Now, build the lightweight layer containing this project's code and RL libraries.

```bash
# 1. Return to this repository
cd ~/workspace/grpo-vision-reasoning

# 2. Build the Environment (Takes ~10 seconds)
docker build -t mi300x-grpo-env .
```

---

## 🚀 Usage

### 1. Launch the Container
Run the container with full access to the MI300X GPUs and host networking.

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

### 2. Configure Your Experiment
Edit `configs/qwen_grpo.yaml` to define your run.

```yaml
run_id: "v1-baseline"           # Unique ID for tracking
learning_rate: 5.0e-6           # LR for the policy
max_steps: 100                  # Training duration
save_steps: 25                  # Checkpoin frequency
hub_model_id: "your-org/model"  # Hugging Face Repo ID to push to
```

### 3. Start Training
Inside the container:

```bash
python3 src/train.py
```

### 4. Monitor
*   **WandB**: Logs will appear under the project `grpo`.
*   **Checkpoints**: Saved locally to `outputs/qwen-vl-grpo/{run_id}`.

### 5. Inference
Run inference on your trained models:

```bash
# Run on a sample image
python3 src/inference/inference.py \
    --run_id v1-baseline \
    --image_path "https://example.com/image.jpg" \
    --prompt "Solve this."

# Run batch inference on all models
python3 src/inference/inference.py --run_id all --prompt "Test prompt"
```

For more details, see [docs/inference_guide.md](docs/inference_guide.md).

---

## 🧠 Reward Functions

The training is guided by functions defined in `src/rewards.py`.
For a detailed explanation of the reward strategy and scoring logic, see [docs/reward_logic.md](docs/reward_logic.md).

*   **`xmlcount_reward_func`**: Encourages strict usage of `<reasoning>` and `<answer>` tags (must appear exactly once).
*   **`soft_format_reward_func`**: Checks if the reasoning comes before the answer.
*   **`strict_format_reward_func`**: Enforces strict newline structure for clean parsing.
*   **`correctness_reward_func`**: Extracts the content within `<answer>` tags and compares it against ground truth.

## 🤝 Contributing

To add a new dataset:
1.  Import it in `src/train.py`.
2.  Update `src/data.py` to map the dataset's columns to the required `prompt`/`answer` format.