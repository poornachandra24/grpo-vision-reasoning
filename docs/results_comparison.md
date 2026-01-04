# RL Training Results: Impact of Reinforcement Learning

This document compares the Base Model (with system prompt) against the RL-Finetuned Model on a complex vision-reasoning task.

**Image**: [GSPO Diagram](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fgit-blob-b3c944808a15dde0a7ff45782f9f074993304bf1%252FCopy%2520of%2520GSPO%2520diagram%2520%281%29.jpg%3Falt%3Dmedia&width=768&dpr=4&quality=100&sign=b546d40a&sv=2)

**Prompt**: "Assuming that all reds = 3, yellow = 2, green = 9, blue = 6 and orange = 7.5. What is logprobs * Advantages? Show your steps and the final answer."

---

## 1. Base Model + System Prompt (No LoRA)
**Run**: `Qwenv3-optimized-mi300x-full-testmini-5` (No LoRA)
**WandB Link**: [Run Link](https://wandb.ai/iris24244242-4good-ai/mi300x-qwen-grpo/runs/kga9ov93) 
**Response (Excerpt)**:
> **<reasoning>**
> ...
> So the resulting matrix is: [[60], [270], [90], [180]]
> ...
> However, the phrasing "Show your steps and the final answer" suggests we might need to present the final answer as a single number...
> If the problem intended the sum, it would be 60 + 270 + 90 + 180 = 600.
> **</reasoning>**
>
> **<answer>**
> 600
> **</answer>**

**Analysis**:
- **Format**: Followed instructions due to system prompt.
- **Confidence**: **Low**. The model second-guessed its correct matrix derivation and forced a scalar sum (`600`) as the final answer.

---

## 2. After RL (Optimized Model)
**Run**: `Qwenv3-optimized-mi300x-full-testmini-5` (RL LoRA)
**WandB Link**: [Run Link]( https://wandb.ai/iris24244242-4good-ai/mi300x-qwen-grpo/runs/uhkbs5l8)

**Response**:
> **<reasoning>**
> ...
> Row 1 (Yellow): sum = 2+2+2+2 = 8 → 8 × 7.5 = 60
> Row 2 (Green): sum = 9+9+9+9 = 36 → 36 × 7.5 = 270
> Row 3 (Red): sum = 3+3+3+3 = 12 → 12 × 7.5 = 90
> Row 4 (Blue): sum = 6+6+6+6 = 24 → 24 × 7.5 = 180
> **</reasoning>**
>
> **<answer>**
> [60, 270, 90, 180]
> **</answer>**

**Analysis**:
- **Format**: Perfect adherence.
- **Confidence**: **High**. The RL training taught the model that the vector itself is the valid answer for this type of question. It did not hesitate or unnecessary convert to a scalar.

---

## Conclusion
Reinforcement Learning did not just teach the model *how* to answer (format), but *what* constitutes a correct answer in this domain (confidence). While the Base Model could follow the system prompt, it lacked the domain-specific conviction that the RL model acquired.
