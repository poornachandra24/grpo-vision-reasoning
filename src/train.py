"""
GRPO Training Script for Qwen2.5-VL on MI300X
Optimized for vLLM 0.11.2 + TRL 0.13.0+
"""

import yaml
import wandb
import os
import sys
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    # Load config
    config_path = "configs/qwen_grpo.yaml"
    logger.info(f"Loading config from {config_path}")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize wandb
    run_id = cfg.get('run_id', 'grpo-run')
    logger.info(f"Initializing WandB project: {cfg['wandb_project']}, run: {run_id}")
    
    wandb.init(
        project=cfg['wandb_project'], 
        name=run_id,
        save_code=True, 
        config=cfg
    )
    
    # 1. Load Model and Processor
    logger.info("Loading model and processor...")
    model, processor = load_model_and_processor(cfg)
    
    # 2. Load and Format Dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        cfg.get('dataset_name', 'AI4Math/MathVista'), 
        split=cfg.get('dataset_split', 'testmini')
    )
    
    # Limit dataset size if specified
    if 'max_samples' in cfg and cfg['max_samples'] > 0:
        logger.info(f"Limiting dataset to {cfg['max_samples']} samples")
        dataset = dataset.select(range(min(cfg['max_samples'], len(dataset))))
    
    logger.info(f"Formatting {len(dataset)} examples...")
    dataset = dataset.map(
        format_data, 
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )
    
    # 3. Configure GRPO Training
    output_dir = f"{cfg['output_dir']}/{run_id}"
    logger.info(f"Output directory: {output_dir}")
    
    training_args = GRPOConfig(
        # Output & Logging
        output_dir=output_dir,
        run_name=run_id,
        logging_steps=cfg.get('logging_steps', 1),
        save_steps=cfg.get('save_steps', 25),
        save_total_limit=cfg.get('save_total_limit', 3),
        report_to="wandb",
        
        # Training Hyperparameters
        num_train_epochs=cfg.get('num_train_epochs', 1),
        max_steps=cfg.get('max_steps', 100),
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg.get('learning_rate', 5e-7),
        
        # Generation Settings
        num_generations=cfg['num_generations'],
        max_prompt_length=cfg['max_seq_length'],
        max_completion_length=cfg.get('max_completion_length', 512),
        temperature=cfg.get('temperature', 1.0),
        top_p=cfg.get('top_p', 0.9),
        
        # Optimization
        bf16=True,
        gradient_checkpointing=True,
        
        # vLLM Configuration
        use_vllm=cfg.get('use_vllm', True),
        vllm_mode=cfg.get('vllm_mode', 'colocate'),
        vllm_gpu_memory_utilization=cfg.get('vllm_gpu_memory_utilization', 0.6),
        vllm_enable_sleep_mode=cfg.get('vllm_enable_sleep_mode', True),
    )
    
    logger.info("GRPO Configuration:")
    logger.info(f"  Batch size: {cfg['batch_size']}")
    logger.info(f"  Gradient accumulation: {cfg['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}")
    logger.info(f"  Max steps: {cfg.get('max_steps', 100)}")
    logger.info(f"  Num generations: {cfg['num_generations']}")
    logger.info(f"  vLLM mode: {cfg.get('vllm_mode', 'colocate')}")
    logger.info(f"  vLLM GPU memory: {cfg.get('vllm_gpu_memory_utilization', 0.6)}")
    
    # 4. Initialize Trainer
    logger.info("Initializing GRPO Trainer...")
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=[
            xmlcount_reward_func, 
            soft_format_reward_func, 
            strict_format_reward_func, 
            correctness_reward_func
        ],
    )
    
    # 5. Train
    logger.info("=" * 60)
    logger.info("Starting GRPO Training")
    logger.info("=" * 60)
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 6. Save Final Model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Push to hub if configured
    if cfg.get('push_to_hub', False):
        hub_model_id = cfg.get('hub_model_id')
        if hub_model_id:
            logger.info(f"Pushing model to hub: {hub_model_id}")
            trainer.push_to_hub(hub_model_id)
        else:
            logger.warning("push_to_hub=True but hub_model_id not specified")
    
    logger.info("Training pipeline complete!")
    wandb.finish()


if __name__ == "__main__":
    main()