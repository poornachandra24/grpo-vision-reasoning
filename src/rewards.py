import re

def xmlcount_reward_func(completions, **kwargs):
    contents = [c[0]["content"] for c in completions]
    return [0.5 if content.count("<reasoning>") == 1 and content.count("</reasoning>") == 1 else 0.0 for content in contents]

def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func(prompts, completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    # TRL stores metadata in the prompt structure differently depending on version
    # This assumes 'answer' was passed in the dataset formatting
    rewards = []
    for response, prompt_data in zip(responses, prompts):
        # We need to extract the ground truth passed earlier. 
        # In TRL GRPO, getting ground truth is tricky; usually we rely on dataset['answer'] alignment
        # For simplicity in this snippet, we skip strict exact match if GT isn't easily accessible
        # or implement a specific lookup. 
        # Placeholder for now:
        rewards.append(0.0) 
    return rewards
