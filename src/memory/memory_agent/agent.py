import logging
import re
import uuid
from datetime import datetime
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.memory.kv_block_manager.block import KVBlock
from src.utils.prompt import MEMORY_AGENT_SYS_PROMPT

logger = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self, model_id: str, 
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   quantization_config=None,
                   max_memory=None,
                   offload_folder=None):
        self.model_id = model_id
        self.model_context_window = model_context_window
        self.block_size = int(model_context_window * 0.9)

        self.summary=None
        
        logger.info(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.debug(f"Tokenizer loaded for {model_id}")

        self.is_active=True     # This will set False if the block is full, and then this agent can not add new memories, but can still be queried
        
        model_kwargs = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = torch.bfloat16
        
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            model_kwargs["offload_folder"] = offload_folder
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        logger.info(f"Model loaded successfully: {model_id}")
        
        # Get layer-to-device mapping
        self.layer_devices = self._get_layer_devices()
        self.primary_device = self.layer_devices.get(0, self.model.device)
        logger.info(f"Layer devices mapped: {len(self.layer_devices)} layers across devices")
        
        self._extract_chat_tokens()
        logger.debug(f"Chat tokens extracted: role_start='{self.role_start}', role_end='{self.role_end}'")
        
        # Initialize first block
        self.current_block = KVBlock(
            block_id=uuid.uuid4(),
            create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            block_size=self.block_size
        )
        self.global_offset = 0
        self.saved_chunks = []  # Store metadata only
        self.chunk_number = 0
        self.merged_cache = None  # Single KV cache in GPU memory
    
    def _get_layer_devices(self):
        """Get accurate device mapping for each layer using hf_device_map."""
        num_layers = len(self.model.model.layers)
        layer_devices = {}
        
        # Use hf_device_map (ground truth for device_map='auto')
        if hasattr(self.model, 'hf_device_map'):
            for key, device_index in self.model.hf_device_map.items():
                if 'model.layers.' in key:
                    try:
                        layer_idx = int(key.split('model.layers.')[1].split('.')[0])
                        if isinstance(device_index, int):
                            layer_devices[layer_idx] = torch.device(f'cuda:{device_index}')
                        else:
                            layer_devices[layer_idx] = torch.device(device_index)
                    except (IndexError, ValueError):
                        continue
        
        # Fallback for missing layers
        for i in range(num_layers):
            if i not in layer_devices:
                try:
                    dev = next(self.model.model.layers[i].parameters()).device
                    layer_devices[i] = torch.device('cuda:0') if dev.type == 'meta' else dev
                except Exception as e:
                    logger.error(f"Error mapping layer {i}: {e}")
                    layer_devices[i] = torch.device('cuda:0')
        
        return layer_devices
    
    def _extract_chat_tokens(self):
        """Extract chat format tokens."""
        test_user = [{"role": "user", "content": "TEST"}]
        test_output = self.tokenizer.apply_chat_template(test_user, tokenize=False, add_generation_prompt=False)
        lines = test_output.split('TEST')
        if len(lines) >= 2:
            before = lines[0].rstrip('\n').split('\n')[-1]
            after = lines[1].split('\n')[0]
            if 'user' in before:
                self.role_start = before.split('user')[0]
                self.role_end = after
            else:
                self.role_start = "<|im_start|>"
                self.role_end = "<|im_end|>"
        else:
            self.role_start = "<|im_start|>"
            self.role_end = "<|im_end|>"
    
    def _add_knowledge(self, text_chunks: List[str]) -> bool:
        """
        Add knowledge chunks incrementally.
        Returns True if block became full, False otherwise.
        Args:
            text_chunks (List[str]): List of text chunks to be added.   
        Returns:
            bool: True if block became full, False otherwise.
        """
        block_full = False
        
        for i, text_chunk in enumerate(text_chunks, 1):
            self.chunk_number += 1
            
            # Format chunk
            if self.global_offset == 0:
                system_msg = [{"role": "system", "content": MEMORY_AGENT_SYS_PROMPT}]
                system_part = self.tokenizer.apply_chat_template(system_msg, tokenize=False, add_generation_prompt=False)
                formatted_chunk = system_part + f"{self.role_start}user\nHere is the context information:\n\n[Context {self.chunk_number}]\n{text_chunk}"
            else:
                formatted_chunk = f"\n\n[Context {self.chunk_number}]\n{text_chunk}"
            
            # Tokenize
            input_ids = self.tokenizer.encode(formatted_chunk, return_tensors="pt", add_special_tokens=False).to(self.primary_device)
            seq_len = input_ids.shape[1]
            
            # Position IDs
            position_ids = torch.arange(self.global_offset, self.global_offset + seq_len, dtype=torch.long, device=self.primary_device).unsqueeze(0)
            
            # Use cached merged KV (incremental update)
            past_kv = self.merged_cache
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_kv, use_cache=True)
            
            # Update merged cache incrementally (single copy)
            self.merged_cache = outputs.past_key_values
            
            # Store only metadata (no cache duplication)
            self.saved_chunks.append({"start": self.global_offset, "length": seq_len})
            self.global_offset += seq_len
            
            # Check if block is full
            block_full = (self.current_block.block_used + seq_len >= self.block_size)
            self.current_block.block_used += seq_len
            
            if block_full:
                logger.info(f"Block {self.current_block.block_id} is full ({self.current_block.block_used}/{self.block_size} tokens)")
                return True
        
        return block_full
    

    def add(self,text_chunks: List[str])->bool:
        """
        Check if the agent is active
        And if so, calling the _add_knowledge function
        Handle the return bool
        If the block is full, set the agent to inactive, and create summaries.
        """
        if not self.is_active:
            raise "The agent is inactive, since the block is already full. So no new knowledge can be added."
        logger.debug(f"Adding {len(text_chunks)} text chunks to memory agent")
        block_full = self._add_knowledge(text_chunks)
        if block_full:
            logger.info("Memory agent became inactive, creating summaries")
            self.is_active = False
            self._create_summaries()
            logger.info("Summaries created successfully")

    def _create_summaries(self):
        """
        Create summaries of all the stored knowledge chunks.
        And this is only needed after the agent is no longer in a active state
        """
        summary_instruction = "Summarize all the context information provided above accurately and concisely."
        self.summary = self._agent_generate(instruction=summary_instruction, max_new_tokens=8192)
        
        # Save cache to disk and clear from GPU
        cache_state = {
            "global_offset": self.global_offset,
            "saved_chunks": self.saved_chunks,
            "chunk_number": self.chunk_number,
            "model_id": self.model_id,
            "merged_cache": [(k.cpu(), v.cpu()) for k, v in self.merged_cache]
        }
        self.current_block.save_cache(cache_state, 0)
        self.merged_cache = None
        torch.cuda.empty_cache()

    def _agent_generate(self,max_new_tokens: int=8192, instruction: str=None, question: str=None)-> str:
        """
        This is a base function for the agent to generate text
        It can be called by query and _create_summaries functions
        Args:
            max_new_tokens (int): Maximum number of tokens to generate.
            instruction (str): The instruction.
            question (str): The user question.
        Returns:
            str: The generated text.
        """
        if not self.saved_chunks:
            logger.warning("No knowledge available for generation")
            return "No knowledge available."
        
        logger.debug(f"Generating response with max_new_tokens={max_new_tokens}")
        
        # Prepare base cache (load from disk if needed)
        base_cache = self.merged_cache
        if base_cache is None:
            cache_state = self.current_block.load_cache()
            if "merged_cache" in cache_state:
                base_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(cache_state["merged_cache"]):
                    target_device = self.layer_devices.get(layer_idx, self.primary_device)
                    base_cache.update(k.to(target_device), v.to(target_device), layer_idx)
            else:
                raise RuntimeError("No cache available for inactive agent")
        
        # CRITICAL: Fork cache to prevent pollution
        generation_cache = DynamicCache()
        
        # Copy cache data (handle different transformers versions)
        for layer_idx in range(len(base_cache)):
            k, v = base_cache[layer_idx]
            generation_cache.update(k, v, layer_idx)
        
        # CRITICAL: Copy _seen_tokens metadata (otherwise model ignores cache)
        if hasattr(base_cache, '_seen_tokens'):
            generation_cache._seen_tokens = base_cache._seen_tokens
        elif len(base_cache) > 0:
            # Fallback: infer from tensor shape
            generation_cache._seen_tokens = base_cache[0][0].shape[-2]
        
        # Format query
        if question:
            # This means we are in the query mode
            formatted_query = f"\n\nBased on the context information provided above, please extract all the original information that is relevant to the question(**REMEBER to give EXACT datetime** along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'.):\n{question}{self.role_end}\n{self.role_start}assistant\n"
        elif instruction:
            formatted_query=f"\n\nBased on the context information provided above, please follow this instruction:\n{instruction}{self.role_end}\n{self.role_start}assistant\n"
        else:
            raise ValueError("Either question or instruction must be provided.")
        
        # Ensure all inputs on first layer device (model input device)
        first_layer_device = self.layer_devices.get(0, self.primary_device)
        input_ids = self.tokenizer.encode(formatted_query, return_tensors="pt", add_special_tokens=False).to(first_layer_device)
        query_len = input_ids.shape[1]
        
        # Position IDs
        query_position_ids = torch.arange(self.global_offset, self.global_offset + query_len, dtype=torch.long, device=first_layer_device).unsqueeze(0)
        
        # Attention mask
        cache_length = generation_cache.get_seq_length()
        if cache_length == 0 and len(generation_cache) > 0:
            # Double check: get from tensor shape
            cache_length = generation_cache[0][0].shape[-2]
        attention_mask = torch.ones((1, cache_length + query_len), dtype=torch.long, device=first_layer_device)
        
        # Manual generation with repetition penalty
        # CRITICAL: Use model.generate() or ensure cache is not mutated
        eos_token_ids = [self.tokenizer.eos_token_id] if not isinstance(self.tokenizer.eos_token_id, (list, tuple)) else self.tokenizer.eos_token_id
        generated_tokens = []
        repetition_penalty = 1.1
        
        # First forward pass with forked cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=query_position_ids,
                past_key_values=generation_cache,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token.item())
            
            # Continue generation with new cache (separate from memory)
            past_kv = outputs.past_key_values
            current_position = self.global_offset + query_len + 1
            next_token_input = next_token
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=first_layer_device)], dim=1)
            
            for _ in range(max_new_tokens - 1):
                if generated_tokens[-1] in eos_token_ids:
                    break
                    
                next_position_ids = torch.tensor([[current_position]], dtype=torch.long, device=first_layer_device)
                outputs = self.model(
                    input_ids=next_token_input,
                    attention_mask=attention_mask,
                    position_ids=next_position_ids,
                    past_key_values=past_kv,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :].clone()
                
                # Repetition penalty
                for token_id in set(generated_tokens[-50:]):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
                
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                generated_tokens.append(next_token.item())
                next_token_input = next_token
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=first_layer_device)], dim=1)
                current_position += 1
                past_kv = outputs.past_key_values
        
        # Decode
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = self._remove_thinking_content(response)
        logger.debug(f"Generated response length: {len(response)} chars")
        
        # Clean up temporary cache
        del generation_cache
        del past_kv
        if not self.is_active and self.merged_cache is not None:
            self.merged_cache = None
        torch.cuda.empty_cache()
        
        return response
    
    def _remove_thinking_content(self, response: str) -> str:
        cleaned_response = response
        # Remove both <thinking> and <think> tags
        cleaned_response = re.sub(r'<thinking>.*?</thinking>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)  # 规范化多个空行
        cleaned_response = cleaned_response.strip()  # 去除首尾空白
        
        return cleaned_response
    
    def query(self, question: str, max_new_tokens: int = 8192) -> str:
        """
        Query using cached knowledge.
        Args:
            question (str): The user question.
            max_new_tokens (int): Maximum number of tokens to generate.
        Returns:
            str: The generated response.
        """
        logger.debug(f"Querying memory agent: {question[:50]}...")
        return self._agent_generate(max_new_tokens=max_new_tokens, question=question)