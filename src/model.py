import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

def load_model_and_processor(config):
    print("Loading Model...")
    processor = AutoProcessor.from_pretrained(config['model_name'])
    
    # Load Base Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        attn_implementation=config['attn_implementation'],
        device_map="auto"
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['lora_target_modules'],
        task_type="CAUSAL_LM",
        lora_dropout=config['lora_dropout']
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor
