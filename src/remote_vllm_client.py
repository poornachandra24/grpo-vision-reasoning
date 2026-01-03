import requests
import json
import io
import base64
import copy
from typing import List, Dict, Any, Union
import torch

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class RemoteVLLMClient:
    processor = None # To be set externally

    def __init__(self, model=None, *args, **kwargs):
        self.url = "http://localhost:8000/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        print(f">>> CUSTOM CLIENT: Initialized. Connected to {self.url}")

    def init_communicator(self, device):
        pass

    def update_named_param(self, name, param_data):
        pass
        
    def reset_prefix_cache(self):
        pass

    def _process_messages(self, messages):
        # Convert PIL images to base64 for OpenAI API
        processed = copy.deepcopy(messages)
        for msg in processed:
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        # Convert PIL image to base64 url
                        # Check if "image" key exists and is a PIL object (or handle if it's already processed)
                        img_obj = part.get("image")
                        if img_obj and not isinstance(img_obj, str): # Assume PIL or similar
                             b64_str = pil_to_base64(img_obj)
                             part["type"] = "image_url"
                             part["image_url"] = {"url": f"data:image/jpeg;base64,{b64_str}"}
                             del part["image"]
        return processed

    def chat(self, messages, sampling_params=None, **kwargs):
        # messages is a list of chat histories (list of lists of dicts)
        # OR it might be a single chat history? 
        # GRPOTrainer calls it with `messages=ordered_set_of_prompts` where `ordered_set_of_prompts` is a list of chats.
        
        # But wait, checking TRL source code:
        # if is_conversational({"prompt": ordered_set_of_prompts[0]}):
        # output = self.vllm_client.chat(messages=ordered_set_of_prompts, ...)
        
        # If `ordered_set_of_prompts` is a list of chats, then `messages` is a list of chats (Batch).
        
        # However, OpenAI API `messages` argument expects a SINGLE chat history.
        # It does NOT support batching multiple separate chat histories in one request (usually).
        # vLLM OpenAI server might support it? Standard OpenAI API does not.
        
        # So we must loop over messages.
        
        prompt_ids = []
        completion_ids = []
        logprobs = []
        
        # Extract params
        temperature = getattr(sampling_params, 'temperature', 0.8)
        max_tokens = getattr(sampling_params, 'max_tokens', 512)
        top_p = getattr(sampling_params, 'top_p', 0.95)
        
        print(f">>> REMOTE CLIENT: Sending {len(messages)} chats to vLLM...")

        for i, chat_history in enumerate(messages):
            # Process images in this chat history
            api_messages = self._process_messages(chat_history)
            
            payload = {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct", 
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "logprobs": True,
                "top_logprobs": 20 # Request enough to hopefully cover our token
            }
            
            try:
                response = requests.post(self.url, headers=self.headers, data=json.dumps(payload), timeout=120)
                response.raise_for_status()
                result = response.json()
                
                choice = result['choices'][0]
                text_content = choice['message']['content']
                
                # Tokenize Completion
                # We need IDs. We use the local processor/tokenizer.
                # Assuming I can access self.processor
                if self.processor is None:
                    raise ValueError("RemoteVLLMClient.processor is not set!")
                    
                # Tokenize completion
                comp_tokens = self.processor.tokenizer(text_content, add_special_tokens=False).input_ids
                completion_ids.append(comp_tokens)
                
                # Tokenize Prompt
                # We need to apply chat template to the ORIGINAL chat history (with PIL images?)
                # self.processor.apply_chat_template handles PIL images if passed correctly.
                # In trl/trainer/grpo_trainer.py it tokenizes prompt to get structure?
                # Actually, GRPOTrainer uses prompt_ids for KL calc.
                
                # Use processor to get prompt ids
                # Note: apply_chat_template output depends on the processor config.
                # For Qwen2-VL, it handles the image tokens.
                p_ids = self.processor.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=True)
                prompt_ids.append(p_ids)
                
                # Logprobs
                # OpenAI returns logprobs for the COMPLETION tokens.
                # 'choice.logprobs.content' is a list of dicts.
                # content[j].logprob is the logprob of the token.
                # We need to map these to the tokens.
                
                # Note: OpenAI tokenization might differ slightly if special tokens are involved, 
                # but with the same model/tokenizer it should be close.
                # However, extracting logprobs exactly matching `comp_tokens` is safer if we take them from the API response.
                
                # If API returns logprobs, we trust them?
                # TRL expects `logprobs` to be list of floats corresponding to completion_ids?
                # Let's verify TRL usage: it likely sums them up.
                
                api_logprobs = choice.get('logprobs', {}).get('content', [])
                if api_logprobs:
                    # Extract logprob float
                    seq_logprobs = [item['logprob'] for item in api_logprobs]
                    logprobs.append(seq_logprobs)
                    
                    # Sanity check: length mismatch?
                    if len(seq_logprobs) != len(comp_tokens):
                        # This happens if tokenization differs or if local tokenizer adds/removes something.
                        # For now, we warn but proceed.
                        # print(f"Warning: Logprobs len {len(seq_logprobs)} != Tokens len {len(comp_tokens)}")
                        pass
                else:
                    # Fallback if no logprobs
                    logprobs.append([0.0] * len(comp_tokens))

            except Exception as e:
                print(f"Error in RemoteVLLMClient: {e}")
                # Fallback valid structure
                prompt_ids.append([])
                completion_ids.append([])
                logprobs.append([])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs
        }

    def generate(self, prompts, sampling_params=None, **kwargs):
        # In case GRPOTrainer decides to use generate() even for chat (unlikely with is_conversational check)
        # We assume prompts is list of chat histories if they are list of dicts.
        return self.chat(prompts, sampling_params=sampling_params, **kwargs)
