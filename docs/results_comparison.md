# RL Fine-Tuning Results: Impact of Reinforcement Learning

This document compares the Base Model (Qwen/Qwen3-VL-8B-Instruct) against the RL-Finetuned Model (Qwen/Qwen3-VL-8B-Instruct + RL-Lora adapters) on the same complex vision-reasoning task on AMD Instinct MI300X.

**Image**: 

- <img width="879" height="765" alt="image" src="https://github.com/user-attachments/assets/4af1c1cf-c322-4af0-a606-e0227ad402d4" />

**System Prompt**:
```
You are a helpful assistant. You must first provide your reasoning in <reasoning> tags, and then your numeric answer in <answer> tags.
```

**User Prompt**:
```
Assuming that all reds = 3, yellow = 2, green = 9, blue = 6 and orange = 7.5. What is logprobs * Advantages(matrix operation)? Show your steps and the final answer.
```
---

## 1. Base Model  (No LoRA adapters)
- **Run**: `Qwenv3-optimized-mi300x-full-testmini-5` (No LoRA) -batch-inf-0104-2012
- **WandB Link**: [Run Link](https://wandb.ai/iris24244242-4good-ai/mi300x-qwen-grpo/runs/kga9ov93) 
**Response**:
```
<reasoning>
We are given a matrix operation: Logprobs ratios * Advantages.

The Logprobs ratios matrix is a 4x4 matrix (4 rows, 4 columns), and the Advantages matrix is a 4x1 matrix (4 rows, 1 column). So, this is a matrix multiplication: (4x4) * (4x1) = (4x1) — a 4x1 result matrix.

We are told the values for each color:
- Red = 3
- Yellow = 2
- Green = 9
- Blue = 6
- Orange = 7.5

We need to interpret the matrices based on the colors shown.

Looking at the Logprobs ratios matrix (left side):
- Row 1: All yellow → [2, 2, 2, 2]
- Row 2: All green → [9, 9, 9, 9]
- Row 3: All red → [3, 3, 3, 3]
- Row 4: All blue → [6, 6, 6, 6]

So the Logprobs ratios matrix is:
[[2, 2, 2, 2],
 [9, 9, 9, 9],
 [3, 3, 3, 3],
 [6, 6, 6, 6]]

The Advantages matrix (right side) has 4 orange boxes, one in each row. Since it's a 4x1 matrix, it should be:
[[7.5],
 [7.5],
 [7.5],
 [7.5]]

Now, we perform matrix multiplication: (4x4) * (4x1) = (4x1)

The result will be a 4x1 matrix, where each element is the dot product of the corresponding row of the first matrix with the column of the second matrix.

Let’s compute each element:

Element 1 (row 1 of Logprobs * column of Advantages):
= (2 * 7.5) + (2 * 7.5) + (2 * 7.5) + (2 * 7.5)
= 4 * (2 * 7.5)
= 4 * 15
= 60

Element 2 (row 2 of Logprobs * column of Advantages):
= (9 * 7.5) + (9 * 7.5) + (9 * 7.5) + (9 * 7.5)
= 4 * (9 * 7.5)
= 4 * 67.5
= 270

Element 3 (row 3 of Logprobs * column of Advantages):
= (3 * 7.5) + (3 * 7.5) + (3 * 7.5) + (3 * 7.5)
= 4 * (3 * 7.5)
= 4 * 22.5
= 90

Element 4 (row 4 of Logprobs * column of Advantages):
= (6 * 7.5) + (6 * 7.5) + (6 * 7.5) + (6 * 7.5)
= 4 * (6 * 7.5)
= 4 * 45
= 180

So, the resulting matrix is:
[[60],
 [270],
 [90],
 [180]]

The question asks for "logprobs * Advantages(matrix operation)" — which we have computed as the 4x1 matrix above.

However, the phrasing "Show your steps and the final answer" suggests we might need to present the final answer as a single number or perhaps the sum of all elements? But the operation is matrix multiplication, and the result is a matrix.

Since the problem says "Show your steps and the final answer", and doesn't specify to sum or do anything else, I think the final answer should be the resulting matrix.

But perhaps the problem expects the sum of all elements of the result matrix? Let's check the context. The problem says "logprobs * Advantages(matrix operation)" — which is matrix multiplication, not sum.

In some contexts, "the result" might mean the sum of the elements, but that's not standard.

Alternatively, maybe the problem is asking for the matrix product, and we need to present it as a matrix.

Given the ambiguity, I will present the matrix as the answer, since that's what matrix multiplication yields.

If the problem intended the sum, it would be 60 + 270 + 90 + 180 = 600.

But I think the correct interpretation is matrix multiplication, yielding a 4x1 matrix.

Looking at the problem again: "What is logprobs * Advantages(matrix operation)?" — it's asking for the result of the operation, which is a matrix.

So I will present the matrix.

But the answer format is numeric, so perhaps they want the sum? Or perhaps the problem is incomplete.

Another thought: maybe the * is not matrix multiplication, but element-wise multiplication? But that would be unusual for matrices of different dimensions (4x4 and 4x1), unless we broadcast.

But the problem says "matrix operation", so matrix multiplication is the most likely.

Given the ambiguity, I will present the matrix as the answer, but since the format requires a numeric answer, perhaps the problem expects the sum.

Let me check the values again.

Alternatively, maybe the "Advantages" matrix is not 4x1, but 1x4? But the image shows 4 orange boxes in a column, so it's 4x1.

Perhaps the problem is asking for the dot product of the two matrices? But that doesn't make sense.

Another possibility: perhaps "logprobs * Advantages" means element-wise multiplication, but that would require the matrices to be the same size, which they are not.

So matrix multiplication is the only sensible operation.

Given the ambiguity, I will assume the problem wants the sum of the elements of the resulting matrix, as that's a common way to interpret "what is the result" when the operation yields a vector.

So let's compute the sum: 60 + 270 + 90 + 180 = 600.

That seems reasonable.

Alternatively, perhaps the problem is asking for the first element? But that doesn't make sense.

Given the problem's phrasing, and the fact that it's a matrix operation, but the answer format is numeric, I think the most likely intent is to sum the elements of the resulting matrix.

So I will go with 600.

But let me double-check my matrix multiplication.

Row 1: [2,2,2,2] * [7.5,7.5,7.5,7.5] = 2*7.5 + 2*7.5 + 2*7.5 + 2*7.5 = 15+15+15+15 = 60 — correct.

Row 2: 9*7.5*4 = 67.5*4 = 270 — correct.

Row 3: 3*7.5*4 = 22.5*4 = 90 — correct.

Row 4: 6*7.5*4 = 45*4 = 180 — correct.

Sum: 60+270+90+180 = 600.

Yes.

Therefore, the final answer is 600.

<answer>600</answer>
```

**Analysis**:
- **Format**: Followed instructions partially due to the system prompt.
- **Confidence**: **Low**. Performed the calculation correctly in the reasoning traces but lacked the confidence to output a matrix. It "hallucinated" a requirement to sum the values ($60+270+90+180=600$), failing the specific request for a matrix operation result.
---

## 2. After RL (Optimized Model)
- **Run**: `Qwenv3-optimized-mi300x-full-testmini-5` (RL LoRA) - batch-inf-0104-1950
- Hf commit: 67e21dbdbeb243569e965957aa42b82df87d039d 
- **WandB Link**: [Run Link]( https://wandb.ai/iris24244242-4good-ai/mi300x-qwen-grpo/runs/uhkbs5l8)

**Response**:
```
<reasoning>
We are given a matrix operation: "Logprobs ratios" matrix multiplied by "Advantages" matrix.

The "Logprobs ratios" matrix is 4x4:
- Row 1: [Yellow, Yellow, Yellow, Yellow] → [2, 2, 2, 2]
- Row 2: [Green, Green, Green, Green] → [9, 9, 9, 9]
- Row 3: [Red, Red, Red, Red] → [3, 3, 3, 3]
- Row 4: [Blue, Blue, Blue, Blue] → [6, 6, 6, 6]

The "Advantages" matrix is 4x1:
- [Orange, Orange, Orange, Orange] → [7.5, 7.5, 7.5, 7.5]

Matrix multiplication (4x4) * (4x1) results in a 4x1 matrix.

Step-by-step multiplication:

First element (Row 1):
= (2 * 7.5) + (2 * 7.5) + (2 * 7.5) + (2 * 7.5)
= 15 + 15 + 15 + 15
= 60

Second element (Row 2):
= (9 * 7.5) + (9 * 7.5) + (9 * 7.5) + (9 * 7.5)
= 67.5 + 67.5 + 67.5 + 67.5
= 270

Third element (Row 3):
= (3 * 7.5) + (3 * 7.5) + (3 * 7.5) + (3 * 7.5)
= 22.5 + 22.5 + 22.5 + 22.5
= 90

Fourth element (Row 4):
= (6 * 7.5) + (6 * 7.5) + (6 * 7.5) + (6 * 7.5)
= 45 + 45 + 45 + 45
= 180

Therefore, the resulting matrix is:
[60]
[270]
[90]
[180]

The question asks for "logprobs * Advantages(matrix operation)", which we have computed as the above 4x1 matrix.
</reasoning>
<answer>
60
270
90
180
</answer>
```



**Analysis**:
- **Format**: Perfect adherence.
- **Confidence**: **High**. Correctly identified the operation, performed the math, and output the vector format without unnecessary reduction


## Inference Graph Analysis
<img width="1473" height="846" alt="image" src="https://github.com/user-attachments/assets/bfb78f4b-56dc-45cf-883c-7c3cdc96778a" />

<img width="1473" height="846" alt="image" src="https://github.com/user-attachments/assets/3f9a641f-a663-42cb-91ca-623cf9d8f4c3" />
<img width="1473" height="846" alt="image" src="https://github.com/user-attachments/assets/180bd643-8299-4a62-9f38-58c2fa03bbad" />

The metrics illustrate a clear distinction between an unoptimized model struggling with "reasoning bloat" and an optimized model executing a streamlined path.


| **Metric Category** | **Specific Metric** | **Base Model (Pink)** | **RL Optimized (Purple)** | **Technical Conclusion** |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Saturation** | `GPU Utilization` | **100%** | **100%** | Both models fully saturate the compute cores. |
| **Memory Traffic** | `memoryReadWriteActivity` | **High (~13)** | **Low (~6)** | **The Critical Difference:** To sustain that 100% compute, the Base model is thrashing the memory subsystem, requiring **2x the bandwidth** of the RL model. |
| **Power Efficiency** | `GPU Power Usage` | **394W** | **320W** | The Base model burns **+74W (+23%)** more power. Since compute is equal (100%), this extra energy is the physical cost of the excessive data movement seen above. |
| **Thermal Load** | `GPU Temperature` | **41°C** | **38°C** | Higher power consumption from inefficient data transport directly results in higher operating temperatures. |
| **Correctness** | *Result Accuracy* | **Incorrect** (Scalar 600) | **Correct** (Vector) | The Base model's confusion led to a "hallucinated" reduction step (summing), likely causing the complex/inefficient reasoning path that spiked memory usage. |

##  Conclusion

The **RL Optimized Model** demonstrates a massive gain in **computational density**.
1.  **Workload:** Both models drove the MI300X to its limit (100% utilization).
2.  **Efficiency:** The RL model achieved this peak performance while cutting memory traffic by **50%** and power by **~20%**.
3.  **Root Cause:** The Base model is **"Memory Thrashing."** It is doing valid compute work on the cores but is architecturally inefficient, constantly fetching and writing data to HBM (High Bandwidth Memory). Moving data is energetically expensive; the **74W power gap** is effectively the "tax" paid for that unoptimized memory access pattern.


