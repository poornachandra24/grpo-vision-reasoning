# specialized Reward Strategy for Reasoning

To ensure the model produces correct answers in the required structured format (Chain-of-Thought), we use a composite reward strategy. This strategy combines **structural rewards** (syntax) with **correctness rewards** (semantics).

## The Objective

We want the model to:
1.  **Think before answering**: Generate a reasoning trace.
2.  **Be Structured**: Use specific XML tags `<reasoning>` and `<answer>`.
3.  **Be Correct**: The final answer must match the ground truth.

## Reward Components

### 1. Structural Rewards (The "Container")
These ensure the model follows the rules of the game.

*   **XML Count (`xmlcount_reward_func`)**
    *   **Goal**: Ensure `<reasoning>` and `<answer>` tags appear **exactly once**.
    *   **Why**: Prevents hallucinating multiple answers or skipping the reasoning step.
    *   **Weight**: `1.0` (Baseline requirement)

*   **Soft Format (`soft_format_reward_func`)**
    *   **Goal**: Ensure correct order: `Reasoning` $\rightarrow$ `Answer`.
    *   **Why**: Enforces "Chain of Thought" behavior.
    *   **Weight**: `0.5`

*   **Strict Format (`strict_format_reward_func`)**
    *   **Goal**: Ensure clean newlines (e.g., `<answer>\nResult\n</answer>`).
    *   **Why**: Makes parsing trivial for downstream evaluation.
    *   **Weight**: `0.3`

### 2. Correctness Reward (The "Content")
This is the most critical signal.

*   **Correctness (`correctness_reward_func`)**
    *   **Goal**: Extract text *inside* the `<answer>` tags and match it to ground truth.
    *   **Why**: A well-formatted wrong answer is useless.
    *   **Weight**: `2.0` (The dominant reward)
    *   **Scoring Logic**:
        *   **+2.0**: Exact Match.
        *   **+1.0**: Partial Match (Answer contained in text).
        *   **0.0**: Incorrect.

## How GRPO Uses These Rewards

GRPO (Group Relative Policy Optimization) generates a group of outputs (e.g., 8 outputs) for the same prompt and sums these rewards for each.

| Output Scenario | Format Score | Correctness Score | Total Reward | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Perfect** (Correct & Structured) | 1.8 | 2.0 | **3.8** | **Highly Reinforced** |
| **Right Answer**, Bad Format | 0.0 | 2.0 | **2.0** | Weak Reinforcement |
| **Wrong Answer**, Perfect Format | 1.8 | 0.0 | **1.8** | Weak Reinforcement |
| **Garbage** | 0.0 | 0.0 | **0.0** | Penalized |

This gradient pushes the model strongly towards the "Perfect" state.

## Configuration

These weights are configurable in `configs/qwen_grpo.yaml`:

```yaml
reward_weights:
  format: 1.0       # xmlcount
  structure: 0.5    # soft_format
  strict: 0.3       # strict_format
  correctness: 2.0  # correctness
```
