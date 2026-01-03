import yaml
import wandb
import torch
import os
from dotenv import load_dotenv
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from src.model import load_model_and_processor
from src.data import format_data
from src.rewards import (
    xmlcount_reward_func, 
    soft_format_reward_func, 
    strict_format_reward_func, 
    correctness_reward_func
)

def main():
    # 0. Load env vars (for local runs or extra safety)
    load_dotenv()

    # 1. Load Config
    config_path = "configs/qwen_grpo.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 2. W&B Init (with run_id in the name for easier tracking)
    wandb_name = f"{cfg['run_id']}-{cfg.get('wandb_run_name', 'run')}"
    wandb.init(
        project=cfg['wandb_project'], 
        name=wandb_name, 
        settings=wandb.Settings(log_code=True),
        config=cfg
    )
    wandb.watch(log_system_metrics=True)

    # 3. Load Resources
    model, processor = load_model_and_processor(cfg)
    
    # Load Data (MathVista mini)
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    # 4. GRPO Configuration (Optimized)
    training_args = GRPOConfig(
        output_dir=f"{cfg['output_dir']}/{cfg['run_id']}", # Unique dir per run
        run_name=wandb_name,
        
        # Optimization
        gradient_checkpointing=True, # Critical for memory efficiency
        bf16=True,
        
        # Training Steps
        max_steps=cfg['max_steps'],
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        
        # GRPO Specifics
        num_generations=cfg['num_generations'],
        max_prompt_length=cfg['max_seq_length'],
        max_completion_length=512,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=cfg['save_steps'],
        save_total_limit=cfg['save_total_limit'], # Prevent filling disk
        logging_steps=cfg.get('logging_steps', 1),
        
        # vLLM Backend
        use_vllm=cfg['use_vllm'],
        vllm_gpu_memory_utilization=cfg['vllm_gpu_memory_utilization'],
        vllm_device="cuda:0",
        vllm_max_model_len=cfg['vllm_max_model_len'],
        
        # Hub Strategy (We push manually at the end, but set token here)
        push_to_hub=False, 
        report_to="wandb",
    )

    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            xmlcount_reward_func, 
            soft_format_reward_func, 
            strict_format_reward_func, 
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    # 6. Train
    print(f"Starting Training Run: {cfg['run_id']}")
    trainer.train()

    # 7. Save & Push to Hub (Custom Logic)
    print("Saving local model...")
    trainer.save_model(training_args.output_dir)
    
    if cfg.get('hub_model_id'):
        print(f"Pushing to Hub: {cfg['hub_model_id']}")
        commit_msg = f"Run ID: {cfg['run_id']} - Trained on MI300X - Steps: {cfg['max_steps']}"
        
        # Push the LoRA adapters
        trainer.push_to_hub(
            repo_id=cfg['hub_model_id'],
            commit_message=commit_msg,
            token=os.environ.get("HF_TOKEN") # Ensure this is in your .env
        )
        print("Push complete!")

    wandb.finish()

if __name__ == "__main__":
    main()
