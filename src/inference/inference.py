"""
Inference CLI for Qwen2.5-VL GRPO Models.
Supports local checkpoints, registry lookup, WandB logging, result persistence,
and batch processing (multiple models x multiple images).
"""
import argparse
import torch
import os
import yaml
import glob
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from datetime import datetime

# Optional: WandB
try:
    import wandb
except ImportError:
    wandb = None

def load_config(config_path="configs/qwen_grpo.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def parse_registry(registry_path="docs/model_registry.md"):
    """Parses the markdown registry table to find runs."""
    if not os.path.exists(registry_path):
        return []
    
    runs = []
    with open(registry_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    for line in lines:
        if "|" in line and "Run ID" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                # Format: | Run ID | Timestamp | HF Commit Hash | Notes |
                runs.append({
                    "run_id": parts[1],
                    "timestamp": parts[2],
                    "commit_hash": parts[3],
                    "notes": parts[4] if len(parts) > 4 else ""
                })
    return runs

def get_latest_local_run(outputs_dir="outputs/qwen-vl-grpo"):
    """Finds the latest modified directory in outputs."""
    if not os.path.exists(outputs_dir):
        return None
    subdirs = [os.path.join(outputs_dir, d) for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return os.path.basename(latest_subdir)

def save_result(run_id, prompt, response, image_path=None, output_file="docs/sample_outputs/results.md"):
    """Persists the result to a markdown file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_str = f"![Image]({image_path})" if image_path else "N/A"
    
    entry = f"""
## Run: {run_id} ({timestamp})
**Prompt:** {prompt}
**Image:** {image_str}

**Response:**
{response}

---
"""
    with open(output_file, "a") as f:
        f.write(entry)
    print(f"Result saved to {output_file}")

def get_image_paths(image_arg):
    """Resolves image argument to a list of paths/URLs."""
    if not image_arg:
        return []
    
    paths = []
    # Check if comma-separated list
    candidates = [p.strip() for p in image_arg.split(",")]
    
    for c in candidates:
        if c.startswith("http"):
             paths.append(c)
        elif os.path.isdir(c):
             # Add all common image extensions
             extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
             for ext in extensions:
                 paths.extend(glob.glob(os.path.join(c, ext)))
                 paths.extend(glob.glob(os.path.join(c, ext.upper())))
        elif os.path.exists(c):
            paths.append(c)
        else:
            print(f"Warning: Image path not found: {c}")
            
    return sorted(list(set(paths)))

def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5-VL GRPO model")
    
    # Model Selection
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_model", type=str, default="Thunderbird2410/grpo-vl-lora")
    parser.add_argument("--run_id", type=str, help="Run ID(s). Comma-separated or 'all'. Defaults to latest registry entry.")
    parser.add_argument("--use_local_only", action="store_true", help="Force using local checkpoint only")
    parser.add_argument("--no_lora", action="store_true", help="Run with base model only (skip adapter loading)")
    
    # Input
    parser.add_argument("--image_path", type=str, help="Path, URL, or Directory of images")
    parser.add_argument("--prompt", type=str, default="Solve this math problem. Show your reasoning.", help="User prompt")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Features
    parser.add_argument("--enable_wandb", action="store_true", help="Log trace to WandB")
    parser.add_argument("--wandb_project", type=str, help="Override WandB project name")
    parser.add_argument("--persist_result_local", action="store_true", help="Save result to docs/sample_outputs")
    
    args = parser.parse_args()
    cfg = load_config()
    runs_registry = parse_registry()
    
    # 1. Resolve Run IDs
    target_runs = []
    
    if args.run_id:
        if args.run_id.lower() == "all":
            target_runs = runs_registry
        else:
            requested_ids = [rid.strip() for rid in args.run_id.split(",")]
            for rid in requested_ids:
                # Find in registry or create dummy if not found (might be local only)
                found = next((r for r in runs_registry if r["run_id"] == rid), {"run_id": rid, "commit_hash": None})
                target_runs.append(found)
    elif runs_registry:
        # Default to latest registry
        target_runs = [runs_registry[-1]]
        print(f"No Run ID specified. Using latest from registry: {target_runs[0]['run_id']}")
    else:
        # Fallback to local
        latest_local = get_latest_local_run(cfg.get("output_dir", "outputs/qwen-vl-grpo"))
        if latest_local:
            target_runs = [{"run_id": latest_local, "commit_hash": None}]
            print(f"Registry empty. Using latest local run: {latest_local}")
            
    if not target_runs:
        print("No valid runs found to execute.")
        return

    # 2. Resolve Images
    image_paths = get_image_paths(args.image_path)
    if not image_paths:
        print("No images found. Running text-only inference (or specify --image_path).")
        image_paths = [None] # Explicit None for text-only loop
        
    print(f"Target Runs: {[r['run_id'] for r in target_runs]}")
    print(f"Target Images: {len(image_paths)}")

    # 3. Execution Loop
    
    # Initialize WandB once if enabled, or per run? 
    # Better to have one WandB run covering the batch, or independent?
    # Let's do a single WandB run for the "Batch Evaluation" session.
    
    if args.enable_wandb and wandb:
        project_name = args.wandb_project or cfg.get("wandb_project", "grpo-vision-reasoning")
        wandb.init(
            project=project_name, 
            name=f"batch-inf-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "base_model": args.base_model,
                "runs": [r['run_id'] for r in target_runs],
                "images": image_paths,
                "system_prompt": args.system_prompt,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        )
        wandb_table = wandb.Table(columns=["Run ID", "Image", "System Prompt", "User Prompt", "Response"])

    # Load Base Model (Load once, reuse)
    print(f"Loading Base Model: {args.base_model}")
    try:
        base_model = AutoModelForVision2Seq.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    for run_info in target_runs:
        run_id = run_info["run_id"]
        commit_hash = run_info.get("commit_hash")
        
        print(f"\n{'='*40}")
        print(f"Processing Run: {run_id}")
        print(f"{'='*40}")

        # Determine Adapter Path
        adapter_path = args.adapter_model
        revision = None
        
        local_path = os.path.join(cfg.get("output_dir", "outputs/qwen-vl-grpo"), run_id)
        
        if os.path.exists(local_path):
            print(f"Found local checkpoint at {local_path}")
            adapter_path = local_path
        elif args.use_local_only:
            print(f"Skipping {run_id}: Local checkpoint missing and --use_local_only is set.")
            continue
        elif commit_hash:
            print(f"Using HF Hub with commit {commit_hash}")
            revision = commit_hash
        else:
            print(f"Warning: No local checkpoint or commit hash for {run_id}. Using latest from Hub.")

        # Load Adapter
        if args.no_lora:
            print(f"Skipping adapter loading for {run_id} (--no_lora set). Using Base Model.")
            model = base_model
        else:
            try:
                # We need to unmerge/unload previous adapter if we want to switch?
                # PeftModel can wrap the base model. To switch, we might need to reload base or use adapters interface.
                # Easiest way for stability: Reload PEFT model on top of base. 
                # Note: PeftModel.from_pretrained returns a new model object wrapping the base.
                # If we reuse 'base_model', we must be careful not to wrap a wrapped model recursively.
                
                # Unbox if it's already a PeftModel
                active_model = base_model
                if isinstance(base_model, PeftModel):
                    active_model = base_model.unload() 
                
                model = PeftModel.from_pretrained(active_model, adapter_path, revision=revision, is_trainable=False)
            except Exception as e:
                print(f"Failed to load adapter for {run_id}: {e}")
                continue

        # Process Images
        for img_path in image_paths:
            print(f"--> Inference on: {img_path if img_path else 'Text Only'}")
            
            # Prepare Input
            content = []
            pil_image = None
            
            if img_path:
                try:
                    if img_path.startswith("http"):
                        pil_image = Image.open(requests.get(img_path, stream=True).raw)
                    else:
                        pil_image = Image.open(img_path)
                    
                    pil_image = pil_image.resize((512, 512))
                    content.append({"type": "image", "image": pil_image})
                except Exception as e:
                    print(f"    Error loading image {img_path}: {e}")
                    continue
            
            content.append({"type": "text", "text": args.prompt})
            
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": args.system_prompt}]})
            
            messages.append({"role": "user", "content": content})
            
            # Inference
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[pil_image] if pil_image else None, padding=True, return_tensors="pt")
                inputs = inputs.to("cuda")
                
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Print concise output
                print(f"    Response: {output[:100]}..." if len(output) > 100 else f"    Response: {output}")
                
                # Persist
                if args.persist_result_local:
                    save_result(run_id, args.prompt, output, img_path)
                
                # Log
                if args.enable_wandb and wandb:
                    wandb_table.add_data(
                        run_id, 
                        wandb.Image(pil_image) if pil_image else (img_path or "Text Only"),
                        args.system_prompt,
                        args.prompt,
                        output
                    )
            except Exception as e:
                print(f"    Inference failed: {e}")

    # Finish WandB
    if args.enable_wandb and wandb:
        wandb.log({"batch_inference_results": wandb_table})
        wandb.finish()

if __name__ == "__main__":
    main()
