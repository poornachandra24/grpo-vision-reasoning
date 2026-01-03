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
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}] 
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": example['question']}
                ]
            }
        ],
        # Force answer to be a string to avoid number/string mix issues
        "answer": str(example['answer']) 
    }
