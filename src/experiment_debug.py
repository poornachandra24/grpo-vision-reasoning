import torch
from transformers import AutoProcessor
from datasets import load_dataset
from data_prep.data import prepare_dataset

def debug_prompt_construction():
    print("="*60)
    print("DEBUGGING PROMPT CONSTRUCTION")
    print("="*60)

    # 1. Load Processor
    import yaml
    with open("configs/qwen_grpo.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    
    print(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 2. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    dataset = dataset.select(range(1)) # Just one example

    # 3. Format Dataset
    print("Formatting dataset...")
    formatted_dataset = prepare_dataset(dataset, processor, num_proc=1)
    
    # 4. Inspect Output
    example = formatted_dataset[0]
    print("\n--- Example 0 ---")
    print(f"Keys: {example.keys()}")
    
    if "prompt" in example:
        print("\n[PROMPT RAW DATA]")
        print(example["prompt"])
        
        # Check System Prompt
        if len(example["prompt"]) > 0 and example["prompt"][0]["role"] == "system":
            system_text = example["prompt"][0]["content"][0]["text"]
            print("\n[SYSTEM PROMPT]")
            print(system_text)
        
        # Check User Prompt
        # With system prompt, user prompt is at index 1
        user_msg_idx = 1 if len(example["prompt"]) > 1 else 0
        text_content = example["prompt"][user_msg_idx]["content"][1]["text"]
        print("\n[INSTRUCTION TEXT]")
        print(text_content)
        
        # Check for tags
        print("\n[TAG CHECK]")
        print(f"Has <reasoning>? { '<reasoning>' in text_content }")
        print(f"Has <answer>? { '<answer>' in text_content }")
    
    print("="*60)

if __name__ == "__main__":
    debug_prompt_construction()
