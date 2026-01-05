# GRPO with Qwen-VL

> **High-Performance Reinforcement Learning Pipeline for Vision-Language Models**  
> Optimized for AMD MI300X Accelerators | Powered by vLLM & TRL

This repository implements **Group Relative Policy Optimization (GRPO)** for fine-tuning **Qwen-VL(2.5 and 3 series)** using reinforcement learning. It is specifically engineered for the **AMD ROCm** ecosystem, leveraging `vLLM` for accelerated rollout generation and LoRA for memory-efficient training.

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
├── docs/                    # Documentation
│   ├── data_prep_decisions.md
│   ├── inference_guide.md
│   ├── model_registry.md
│   ├── reward_logic.md
│   └── sample_outputs/
├── src/
│   ├── data_prep/           # Data cleaning and formatting
│   │   └── data.py
│   ├── fine_tuning/         # Training logic
│   │   ├── model.py         # Model loading with QLoRA/LoRA adapters
│   │   ├── rewards.py       # Custom reward functions
│   │   └── train.py         # Main entry point: Trainer setup, MLOps, & Execution
│   └── inference/           # Inference tools
│       └── inference.py
├── build_vllm_mi300x.sh     # vLLM Base Image Build Script
├── Dockerfile               # Production environment definition (Stage 2)
└── requirements.txt         # Python dependencies
```

---

## 🛠️ Installation & Build

We use a **Two-Stage Build Strategy** to ensure stability with ROCm dependencies.

### Stage 1: Build Base vLLM Image
First, we build the official vLLM image from source. We provide a script to handle this automatically.

```bash
# 1. Run the build script
bash build_vllm_mi300x.sh
```

### Stage 2: Build Project Image
Now, build the lightweight layer containing this project's code and RL libraries.

```bash
# 1. Build the Environment (Takes ~10 seconds)
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
    mi300x-grpo-env:v0.11.2
```

### 2. Configure Your Experiment
Edit `configs/qwen_grpo.yaml` to define your run.

```yaml
run_id: "v1-baseline"           # Unique ID for tracking
output_dir: "outputs/qwen-vl-grpo"
model_name: "Qwen/Qwen2.5-VL-7B-Instruct"

# Training
learning_rate: 5.0e-6
max_steps: 100
batch_size: 8
gradient_accumulation_steps: 2

# Generation
num_generations: 8
max_completion_length: 512

# Dataset
dataset_name: "AI4Math/MathVista"
dataset_split: "testmini"

# vLLM (Critical for Performance)
use_vllm: true
vllm_gpu_memory_utilization: 0.7
vllm_enable_sleep_mode: true    # Important for preventing OOM
```

### 3. Start Training
Inside the container:

```bash
python3 src/fine_tuning/train.py
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

The training is guided by functions defined in `src/fine_tuning/rewards.py`.
For a detailed explanation of the reward strategy and scoring logic, see [docs/reward_logic.md](docs/reward_logic.md).

*   **`xmlcount_reward_func`**: Encourages strict usage of `<reasoning>` and `<answer>` tags (must appear exactly once).
*   **`soft_format_reward_func`**: Checks if the reasoning comes before the answer.
*   **`strict_format_reward_func`**: Enforces strict newline structure for clean parsing.
*   **`correctness_reward_func`**: Extracts the content within `<answer>` tags and compares it against ground truth.


## Results

We have successfully trained the model to follow a strict thought-process format (`<reasoning>` and `<answer>` tags) and perform complex visual reasoning.

- **Before RL**: The model failed to follow format instructions and often hallucinated on matrix logical tasks.
- **After RL**: The model correctly reasons through multi-step matrix operations and faithfully outputs the required XML tags.

See [docs/results_comparison.md](docs/results_comparison.md) for a detailed Before/After comparison.



## References:
1. https://huggingface.co/docs/trl/main/en/grpo_trainer
2. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
3. https://huggingface.co/docs/trl/main/en/vllm_integration
4. https://docs.vllm.ai/en/stable/training/trl/

   




