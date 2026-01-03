"""
Inference CLI for Qwen2.5-VL GRPO Models.
"""
import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import requests
import os

def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5-VL GRPO model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_model", type=str, default="Thunderbird2410/grpo-vl-lora")
    parser.add_argument("--commit_hash", type=str, help="Specific HF commit hash to load adapter from")
    parser.add_argument("--image_path", type=str, help="Path to image for inference")
    parser.add_argument("--prompt", type=str, default="Solve this math problem.", help="User prompt")
    
    args = parser.parse_args()
    
    print(f"Loading Base Model: {args.base_model}")
    print(f"Adapter: {args.adapter_model} (Commit: {args.commit_hash or 'Latest'})")
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load Base Model
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load Adapter
    if args.commit_hash:
        print(f"Loading adapter from specific commit: {args.commit_hash}...")
        model = PeftModel.from_pretrained(
            model, 
            args.adapter_model, 
            revision=args.commit_hash,
            is_trainable=False
        )
    else:
        print(f"Loading latest adapter...")
        model = PeftModel.from_pretrained(
            model, 
            args.adapter_model,
            is_trainable=False
        )
        
    print("Model loaded successfully.")
    
    # Prepare Input
    content = []
    images = []
    
    if args.image_path:
        if args.image_path.startswith("http"):
            image = Image.open(requests.get(args.image_path, stream=True).raw)
        else:
            image = Image.open(args.image_path)
            
        # Resize to standard if needed (optional, but good for consistency)
        image = image.resize((512, 512))
        images.append(image)
        content.append({"type": "image", "image": image})
    
    content.append({"type": "text", "text": args.prompt})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt"
    )
        
    inputs = inputs.to("cuda")
    
    # Generate
    print("Generating...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("OUTPUT")
    print("="*60)
    # The output usually contains the prompt too, extracting new tokens might be better but raw is fine
    print(output_text[0])
    print("="*60)

if __name__ == "__main__":
    main()
