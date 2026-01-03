import re

# 1. XML Count: Reward if tags appear exactly once
def xmlcount_reward_func(completions, **kwargs):
    contents = [c[0]["content"] for c in completions]
    return [0.5 if content.count("<reasoning>") == 1 and content.count("</reasoning>") == 1 else 0.0 for content in contents]

# 2. Soft Format: Reward if the structure looks roughly right (regex match)
def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 3. Strict Format: Reward for specific newline structure (The Notebook uses this too)
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 4. Correctness: Checks the math/vision answer against ground truth
def correctness_reward_func(prompts, completions, answer, **kwargs):
    # Note: The 'answer' argument comes from the dataset column 'answer'
    # TRL automatically passes extra dataset columns as kwargs to reward functions.
    
    responses = [c[0]["content"] for c in completions]
    rewards = []
    
    for response, ground_truth in zip(responses, answer):
        # Extract the content inside <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
            # Check for exact match (Vision tasks usually require specific strings)
            # You can loosen this to 'in' or float comparison if doing pure math
            if extracted_answer == ground_truth:
                rewards.append(2.0) # High reward for getting it right
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0) # No answer found
            
    return rewards