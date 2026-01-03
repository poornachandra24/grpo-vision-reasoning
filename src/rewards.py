"""
Reward Functions for GRPO Training
Designed for structured reasoning format with <reasoning> and <answer> tags
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_completion_text(completions):
    """
    Extract text from TRL completion format.
    
    TRL passes completions in different formats depending on version.
    This handles both common formats:
    - List[str] - direct strings
    - List[List[dict]] - message format with role/content
    """
    results = []
    
    for completion in completions:
        if isinstance(completion, str):
            # Direct string format
            results.append(completion)
        elif isinstance(completion, list) and len(completion) > 0:
            # Message format - extract content from last message
            if isinstance(completion[0], dict) and "content" in completion[0]:
                results.append(completion[0]["content"])
            else:
                # Fallback
                results.append(str(completion))
        else:
            # Unknown format
            results.append(str(completion))
    
    return results


# 1. XML Count: Reward if tags appear exactly once
def xmlcount_reward_func(completions, **kwargs):
    """
    Rewards completions that have exactly one <reasoning> block and one <answer> block.
    
    Returns:
        0.5 if both tags appear exactly once
        0.0 otherwise
    """
    contents = extract_completion_text(completions)
    
    rewards = []
    for content in contents:
        has_reasoning = content.count("<reasoning>") == 1 and content.count("</reasoning>") == 1
        has_answer = content.count("<answer>") == 1 and content.count("</answer>") == 1
        
        reward = 0.5 if (has_reasoning and has_answer) else 0.0
        rewards.append(reward)
    
    return rewards


# 2. Soft Format: Reward if the structure looks roughly right (regex match)
def soft_format_reward_func(completions, **kwargs):
    """
    Rewards completions that have reasoning followed by answer (flexible whitespace).
    
    Returns:
        0.5 if pattern matches
        0.0 otherwise
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    contents = extract_completion_text(completions)
    
    rewards = []
    for content in contents:
        match = re.search(pattern, content, re.DOTALL)
        reward = 0.5 if match else 0.0
        rewards.append(reward)
    
    return rewards


# 3. Strict Format: Reward for specific newline structure
def strict_format_reward_func(completions, **kwargs):
    """
    Rewards completions with strict formatting:
    <reasoning>
    [content]
    </reasoning>
    <answer>
    [content]
    </answer>
    
    Returns:
        0.5 if strict format matches
        0.0 otherwise
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    contents = extract_completion_text(completions)
    
    rewards = []
    for content in contents:
        match = re.search(pattern, content, re.DOTALL)
        reward = 0.5 if match else 0.0
        rewards.append(reward)
    
    return rewards


# 4. Correctness: Checks answer against ground truth
def correctness_reward_func(completions, answer=None, **kwargs):
    """
    Rewards completions that have the correct answer inside <answer> tags.
    
    Args:
        completions: List of generated completions
        answer: List of ground truth answers (from dataset)
        
    Returns:
        2.0 if answer is correct
        0.0 otherwise
    """
    contents = extract_completion_text(completions)
    
    # Handle case where answer is not provided
    if answer is None:
        logger.warning("No ground truth answers provided to correctness_reward_func")
        return [0.0] * len(contents)
    
    # Ensure answer is a list
    if not isinstance(answer, list):
        answer = [answer] * len(contents)
    
    rewards = []
    for content, ground_truth in zip(contents, answer):
        # Extract content inside <answer> tags
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        
        if match:
            extracted_answer = match.group(1).strip()
            ground_truth_str = str(ground_truth).strip()
            
            # Check for exact match (case-insensitive)
            if extracted_answer.lower() == ground_truth_str.lower():
                rewards.append(2.0)  # High reward for correct answer
            else:
                # Check if answer is contained (partial credit)
                if ground_truth_str.lower() in extracted_answer.lower():
                    rewards.append(1.0)  # Partial reward
                else:
                    rewards.append(0.0)
        else:
            # No answer tag found
            rewards.append(0.0)
    
    return rewards


# 5. Combined Reward (Optional - use this if you want a single weighted reward)
def combined_reward_func(completions, answer=None, **kwargs):
    """
    Combines all reward functions with weights.
    
    Weights:
        - Format (xmlcount): 1.0
        - Structure (soft format): 0.5
        - Strict formatting: 0.3
        - Correctness: 2.0
        
    Returns:
        Weighted sum of all rewards
    """
    # Get individual rewards
    format_rewards = xmlcount_reward_func(completions, **kwargs)
    structure_rewards = soft_format_reward_func(completions, **kwargs)
    strict_rewards = strict_format_reward_func(completions, **kwargs)
    correctness_rewards = correctness_reward_func(completions, answer=answer, **kwargs)
    
    # Combine with weights
    weights = {
        'format': 1.0,
        'structure': 0.5,
        'strict': 0.3,
        'correctness': 2.0
    }
    
    combined_rewards = []
    for i in range(len(completions)):
        total_reward = (
            weights['format'] * format_rewards[i] +
            weights['structure'] * structure_rewards[i] +
            weights['strict'] * strict_rewards[i] +
            weights['correctness'] * correctness_rewards[i]
        )
        combined_rewards.append(total_reward)
    
    return combined_rewards


# Utility function for debugging rewards
def debug_rewards(completions, answer=None, **kwargs):
    """
    Print detailed reward breakdown for debugging.
    """
    contents = extract_completion_text(completions)
    
    print("\n" + "="*60)
    print("REWARD DEBUG")
    print("="*60)
    
    for i, content in enumerate(contents):
        print(f"\n--- Completion {i+1} ---")
        print(f"Content: {content[:200]}...")  # First 200 chars
        print(f"\nRewards:")
        print(f"  XML Count: {xmlcount_reward_func([completions[i]], **kwargs)[0]}")
        print(f"  Soft Format: {soft_format_reward_func([completions[i]], **kwargs)[0]}")
        print(f"  Strict Format: {strict_format_reward_func([completions[i]], **kwargs)[0]}")
        
        if answer is not None:
            answer_val = answer[i] if isinstance(answer, list) else answer
            print(f"  Correctness: {correctness_reward_func([completions[i]], answer=[answer_val], **kwargs)[0]}")
            print(f"  Ground Truth: {answer_val}")
    
    print("\n" + "="*60 + "\n")