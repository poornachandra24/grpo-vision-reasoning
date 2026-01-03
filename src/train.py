import yaml
import wandb
import os
import sys
import logging
import requests
import json

# --- MONKEY PATCH: CUSTOM REMOTE CLIENT ---
from trl.trainer import grpo_trainer
from src.remote_vllm_client import RemoteVLLMClient

# Inject the custom client
grpo_trainer.VLLMClient = RemoteVLLMClient
# ------------------------------------------

# Setup Logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from src.model import load_model_and_processor
from src.data import format_data
from src.rewards import (
    xmlcount_reward_func, soft_format_reward_func, 
    strict_format_reward_func, correctness_reward_func
)

def main():
    config_path = "configs/qwen_grpo.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg['wandb_project'], name=f"{cfg.get('run_id','run')}-custom-client", save_code=True, config=cfg)
    
    # 1. Load Training Model
    print(">>> Loading Training Model...")
    model, processor = load_model_and_processor(cfg)
    RemoteVLLMClient.processor = processor
    
    # 2. Load Data
    print(">>> Loading Data...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    # 3. Config
    training_args = GRPOConfig(
        output_dir=f"{cfg['output_dir']}/{cfg.get('run_id', 'default')}",
        run_name=f"{cfg.get('run_id','run')}-custom-client",
        gradient_checkpointing=True,
        bf16=True,
        max_steps=cfg.get('max_steps', 100),
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        num_generations=cfg['num_generations'],
        max_prompt_length=cfg['max_seq_length'],
        max_completion_length=512,
        save_steps=25,
        logging_steps=1,
        report_to="wandb",
        
        # --- CONFIG HACK ---
        # We enable use_vllm to trigger the GRPOTrainer to use our injected client.
        use_vllm=True,
        vllm_gpu_memory_utilization=0.4, 
    )

    # 4. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    print(">>> Starting Training (Using Custom vLLM Client)...")
    trainer.train()
    
    print(">>> Saving...")
    trainer.save_model(training_args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()