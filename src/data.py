# The Unsloth Notebook System Prompt
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

def format_data(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]},
        ],
        # We explicitly keep the 'answer' column.
        # TRL's GRPOTrainer will pass this list to the reward function.
        "answer": example['answer'] 
    }