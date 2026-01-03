import yaml
import os
import wandb
from dotenv import load_dotenv
import torch
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from src.model import load_model_and_processor
from src.data import format_data
from src.rewards import xmlcount_reward_func, soft_format_reward_func

def main():
    load_dotenv()
    with open("configs/qwen_grpo.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg['wandb_project'], name=cfg['wandb_run_name'], settings=wandb.Settings(log_code=True))
    wandb.watch(log_system_metrics=True)

    model, processor = load_model_and_processor(cfg)
    
    # Load MathVista (Mini test set for speed)
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    training_args = GRPOConfig(
        output_dir=cfg['output_dir'],
        learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        num_generations=cfg['num_generations'],
        max_prompt_length=cfg['max_seq_length'],
        max_completion_length=512,
        num_train_epochs=1,
        max_steps=cfg['steps'],
        bf16=True,
        report_to="wandb",
        use_vllm=cfg['use_vllm'],
        vllm_gpu_memory_utilization=cfg['vllm_gpu_memory_utilization'],
        vllm_device="cuda:0",
        vllm_max_model_len=cfg['vllm_max_model_len'],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[xmlcount_reward_func, soft_format_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model(cfg['output_dir'])

if __name__ == "__main__":
    main()
