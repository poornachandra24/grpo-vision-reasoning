"""
Model and Processor Loading for Qwen2.5-VL
Optimized for MI300X with LoRA
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)


def load_model_and_processor(config):
    """
    Load Qwen2.5-VL model with LoRA adapters.
    
    IMPORTANT: Qwen2.5-VL uses a different architecture than Qwen2-VL.
    We must use AutoModelForVision2Seq, not Qwen2VLForConditionalGeneration.
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        model: PEFT-wrapped model ready for training
        processor: Qwen2.5-VL processor for image + text
    """
    model_name = config['model_name']
    logger.info(f"Loading model: {model_name}")
    
    # 1. Load Processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 2. Load Model - Use AutoModelForVision2Seq for Qwen2.5-VL
    logger.info("Loading base model...")
    
    # Use Auto class which will load the correct architecture
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get('attn_implementation', 'flash_attention_2'),
        device_map=None,  # Manual control for vLLM compatibility
        trust_remote_code=True
    )
    
    # Manually move to GPU
    logger.info("Moving model to GPU...")
    model = model.to("cuda")
    
    # 3. Apply LoRA
    logger.info("Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        target_modules=config.get('lora_target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=config.get('lora_dropout', 0.05),
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    logger.info("=" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Architecture: {model.config.model_type}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable: {trainable_pct:.2f}%")
    logger.info(f"  LoRA rank: {config.get('lora_r', 16)}")
    logger.info(f"  LoRA alpha: {config.get('lora_alpha', 32)}")
    logger.info("=" * 60)
    
    # Enable gradient checkpointing to save memory
    if config.get('gradient_checkpointing', True):
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    return model, processor