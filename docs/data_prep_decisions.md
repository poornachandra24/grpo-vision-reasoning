# Data Preparation Decisions for Qwen2.5-VL GRPO

This document records the architectural and implementation decisions for the data processing pipeline used in GRPO training.

## 1. Image Processing
- **Resolution**: All images are resized to **512x512**. This standardizes input size for the vision encoder and helps manage memory usage, consistent with Unsloth's Qwen2.5-VL sample code.
- **Format**: Images are converted to **RGB** mode.
- **Decoding**: We cast the `image` column to the `datasets.Image()` feature. This ensures that when the dataset is accessed, the image data is correctly decoded into PIL Image objects, which the processor expects.

## 2. Filtering
- **Numeric Answers**: We filter the dataset to include only examples where the answer can be parsed as a float (`is_numeric_answer`). This is essential for the math reasoning task where reward functions often verify numerical correctness.

## 3. Prompt Engineering
- **Multimodal Structure**: 
    - We provide the prompt as a **list of messages** (dictionaries), specifically:
      ```python
      [
          {
              "role": "user", 
              "content": [
                  {"type": "image"}, 
                  {"type": "text", "text": "..."}
              ]
          }
      ]
      ```
    - This structure is required by `GRPOTrainer` to correctly handle multimodal inputs.
- **Reasoning Enforcement**: The user prompt is augmented with instructions to enclose reasoning in `<REASONING>...</REASONING>` tags and the final answer in `<SOLUTION>...</SOLUTION>` tags. This facilitates the parsing logic required by the GRPO reward functions.

## 4. Pipeline Integration
- **No Pre-templating**: We explicitly **do not** apply `tokenizer.apply_chat_template` during the map phase. 
    - *Rationale*: Applying the template converts the structured list into a single string. `GRPOTrainer` expects the raw list structure for multimodal models to properly insert image tokens and handle the vision encoder inputs. Pre-templating causes `TypeError` during training.

## 5. Performance Optimization
- **Parallel Processing**: 
    - We utilize the `num_proc` argument in Hugging Face Datasets `map` and `filter` operations.
    - Controlled via `dataset_num_proc` in the configuration (default: 16).
    - *Rationale*: Data preparation involves CPU-bound tasks like image resizing and format conversion. Parallelizing these operations significantly reduces the startup time before training begins, minimizing GPU idle time.
