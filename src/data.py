"""
Dataset Formatting for Qwen2.5-VL GRPO Training
Formats MathVista and similar vision-math datasets
"""

import logging

logger = logging.getLogger(__name__)

# System prompt that enforces structured reasoning format
SYSTEM_PROMPT = """You are a helpful AI assistant that solves mathematical problems step-by-step.

Always respond in the following XML format:

<reasoning>
[Explain your step-by-step reasoning here]
</reasoning>

<answer>
[Provide the final answer here]
</answer>

Be clear, concise, and show your work."""


def format_data(example):
    """
    Format a single example for GRPO training.
    
    Expected input fields:
        - decoded_image: PIL Image or image data
        - question: str, the math problem
        - answer: str/int/float, ground truth answer
        
    Returns:
        dict with:
            - prompt: List of message dicts for Qwen2.5-VL
            - answer: str, ground truth for reward calculation
    """
    
    # Extract fields
    image = example.get('decoded_image') or example.get('image')
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    # Ensure answer is a string for consistent reward calculation
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Build prompt in Qwen2.5-VL format
    prompt = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    return {
        "prompt": prompt,
        "answer": answer
    }


def format_data_text_only(example):
    """
    Format a text-only example (for datasets without images).
    
    Use this for pure math problems or when testing without vision.
    """
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    if not isinstance(answer, str):
        answer = str(answer)
    
    prompt = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question}
            ]
        }
    ]
    
    return {
        "prompt": prompt,
        "answer": answer
    }


def validate_formatted_data(dataset):
    """
    Validate that dataset is properly formatted.
    
    Args:
        dataset: HuggingFace Dataset after formatting
        
    Raises:
        ValueError if formatting issues detected
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")
    
    sample = dataset[0]
    
    # Check required fields
    if 'prompt' not in sample:
        raise ValueError("Dataset missing 'prompt' field")
    if 'answer' not in sample:
        raise ValueError("Dataset missing 'answer' field")
    
    # Check prompt format
    prompt = sample['prompt']
    if not isinstance(prompt, list):
        raise ValueError("Prompt should be a list of messages")
    
    if len(prompt) < 2:
        raise ValueError("Prompt should have at least system + user messages")
    
    # Check message format
    for msg in prompt:
        if 'role' not in msg or 'content' not in msg:
            raise ValueError("Each message should have 'role' and 'content'")
    
    logger.info("✓ Dataset validation passed")
    logger.info(f"  Total examples: {len(dataset)}")
    logger.info(f"  Sample prompt roles: {[m['role'] for m in prompt]}")
    logger.info(f"  Sample answer type: {type(sample['answer'])}")