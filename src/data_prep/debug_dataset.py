import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset
from transformers import AutoProcessor
from data import prepare_dataset
import logging

logging.basicConfig(level=logging.INFO)

print("Loading processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

print("Loading dataset...")
dataset = load_dataset("AI4Math/MathVista", split="testmini")
# Take a small subset for debugging
dataset = dataset.select(range(5))

print("Preparing dataset...")
processed_dataset = prepare_dataset(dataset, processor, num_proc=2)

print("\nValidation Results:")
print(f"Original size: {len(dataset)}")
print(f"Processed size: {len(processed_dataset)}")

if len(processed_dataset) > 0:
    example = processed_dataset[0]
    print("\nKeys:", example.keys())
    print("\nPrompt Type:", type(example["prompt"]))
    if isinstance(example["prompt"], list):
        print("Prompt (First message):", example["prompt"][0])
    else:
        print("Prompt (First 500 chars):", example["prompt"][:500])
    print("\nImage:", example["image"])
    print("Image Size:", example["image"].size)
    print("Answer:", example["answer"])
else:
    print("Dataset empty after filtering!")
