SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

def format_data(example):
    # Mapping for MathVista or similar datasets
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]},
        ],
        "answer": example['answer'] 
    }
