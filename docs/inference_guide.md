# Qwen2.5-VL GRPO Inference Guide

The project includes a robust inference CLI located at `src/inference/inference.py`.

## Core Features
1.  **Model Registry Integration**: The script parses `docs/model_registry.md` to find model checkpoints by `Run ID`.
2.  **Automatic Fallback**: It searches locally first, then attempts to download from Hugging Face Hub if a commit hash is available.
3.  **WandB Logging**: Can log inputs (text + images) and outputs to a Weights & Biases Table.
4.  **Batch Processing**: Supports running multiple models against multiple images in a single command.
5.  **Local Persistence**: Optionally saves results to markdown files for easy viewing.

## Usage

### Basic Usage
```bash
python3 src/inference/inference.py \
    --run_id v2-optimized-mi300x \
    --image_path "path/to/image.jpg" \
    --system_prompt "You are a visual reasoning expert." \
    --prompt "Describe this image."
```
*   If `--run_id` is omitted, it defaults to the latest entry in the registry.
*   `--image_path` accepts local paths or HTTP(S) URLs.

### Advanced Options

#### Batch Inference
Run specific models against a list of images:
```bash
python3 src/inference/inference.py \
    --run_id "v2-optimized-mi300x,v1-baseline" \
    --image_path "img1.jpg,img2.jpg,https://example.com/img3.png"
```

Run **ALL** models in the registry:
```bash
python3 src/inference/inference.py \
    --run_id all \
    --prompt "Evaluate this math problem."
```

#### Logging & Persistence
Enable WandB and local saving:
```bash
python3 src/inference/inference.py \
    --run_id v2-optimized-mi300x \
    --enable_wandb \
    --persist_result_local
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--run_id` | Run ID(s) to load. Comma-separated or 'all'. | Latest from Registry |
| `--image_path` | Path/URL to image(s). Comma-separated or directory. | None |
| `--prompt` | User prompt text. | "Solve this math problem..." |
| `--system_prompt` | System prompt text. | "You are a helpful assistant." |
| `--max_new_tokens` | Maximum generation length. | 1024 |
| `--temperature` | Sampling temperature. | 0.7 |
| `--top_p` | Nucleus sampling probability. | 0.9 |
| `--enable_wandb` | Log trace to WandB. | False |
| `--persist_result_local` | Save result to `docs/sample_outputs/results.md`. | False |
| `--use_local_only` | Force using local checkpoint only (error if missing). | False |
| `--no_lora` | Run with base model only (skip adapter loading). | False |
