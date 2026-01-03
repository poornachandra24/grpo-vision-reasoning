"""
Dataset Formatting for Qwen2.5-VL GRPO Training
"""

import logging
from PIL import Image
from datasets import Image as ImageFeature

logger = logging.getLogger(__name__)

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except:
        return False

def resize_images(example):
    image = example["decoded_image"]
    # Resize to (512, 512) as per sample code for consistency/memory
    image = image.resize((512, 512))
    example["decoded_image"] = image
    return example

def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example

def make_conversation(example):
    # The user's text prompt with reasoning instructions
    text_content = (
        f"{example['question']}. Also first provide your reasoning or working out"\
        f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
    )

    # Construct the prompt properly for the processor
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": text_content},
            ],
        },
    ]
    
    # Return separated fields suitable for map
    return {
        "prompt": prompt, 
        "image": example["decoded_image"], 
        "answer": example["answer"]
    }

def prepare_dataset(dataset, tokenizer):
    """
    Full pipeline to prepare the dataset:
    1. Filter numeric answers
    2. Resize images
    3. Convert to RGB
    4. Create conversation format
    5. Clean columns
    6. Apply chat template
    """
    logger.info("Filtering for numeric answers...")
    dataset = dataset.filter(is_numeric_answer)
    
    logger.info("Resizing images to 512x512...")
    dataset = dataset.map(resize_images)
    
    logger.info("Converting images to RGB...")
    dataset = dataset.map(convert_to_rgb)
    
    logger.info("Formatting conversations...")
    dataset = dataset.map(make_conversation)
    
    # Remove original image column and rename decoded_image if needed
    # (Based on sample code logic, but here make_conversation already returns 'image')
    # Remove original image column and rename decoded_image if needed
    # (Based on sample code logic, but here make_conversation already returns 'image')
    # Use select_columns to keep only what we need to avoid legacy column issues
    dataset = dataset.select_columns(["prompt", "image", "answer"])

    # Cast image column to Image feature to ensure it decodes to PIL.Image
    dataset = dataset.cast_column("image", ImageFeature())
    
    return dataset