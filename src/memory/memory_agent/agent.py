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
        
        # Get available devices for cache distribution
        if hasattr(self.model, 'hf_device_map'):
            # Multi-GPU: extract all GPU devices
            unique_devices = set()
            for device in self.model.hf_device_map.values():
                if isinstance(device, str) and 'cuda' in device:
                    unique_devices.add(torch.device(device))
                elif isinstance(device, torch.device) and device.type == 'cuda':
                    unique_devices.add(device)
            self.available_devices = sorted(list(unique_devices), key=lambda d: d.index)
            self.primary_device = self.available_devices[0] if self.available_devices else self.model.device
        else:
            # Single device
            self.primary_device = self.model.device
            self.available_devices = [self.primary_device]
        logger.info(f"Using {len(self.available_devices)} device(s) for cache: {self.available_devices}")
        
        self._extract_chat_tokens()
        logger.debug(f"Chat tokens extracted: role_start='{self.role_start}', role_end='{self.role_end}'")
        
        # Initialize first block
        self.current_block = KVBlock(
            block_id=uuid.uuid4(),
            create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            block_size=self.block_size
        )
        self.global_offset = 0
        self.saved_chunks = []
        self.chunk_number = 0
    
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
            
            # Load previous cache
            past_kv = None
            if self.saved_chunks:
                past_kv = DynamicCache()
                num_layers = len(self.saved_chunks[0]["cache"])
                for layer_idx in range(num_layers):
                    keys, values = [], []
                    target_device = self.available_devices[layer_idx % len(self.available_devices)]
                    for chunk_info in self.saved_chunks:
                        k, v = chunk_info["cache"][layer_idx]
                        keys.append(k.to(target_device, non_blocking=True))
                        values.append(v.to(target_device, non_blocking=True))
                    past_kv.update(torch.cat(keys, dim=2), torch.cat(values, dim=2), layer_idx)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_kv, use_cache=True)
            
            # Extract new cache only and distribute across GPUs
            full_kv_cache = outputs.past_key_values
            new_cache_only = []
            num_layers = len(full_kv_cache)
            for layer_idx in range(num_layers):
                k_full, v_full = full_kv_cache[layer_idx]
                k_new = k_full[:, :, -seq_len:, :]
                v_new = v_full[:, :, -seq_len:, :]
                # Distribute layers across available devices
                target_device = self.available_devices[layer_idx % len(self.available_devices)]
                new_cache_only.append((k_new.to(target_device), v_new.to(target_device)))
            
            # Store chunk
            self.saved_chunks.append({"cache": new_cache_only, "start": self.global_offset, "length": seq_len})
            self.global_offset += seq_len
            
            # Save to block
            cache_state = {
                "global_offset": self.global_offset,
                "saved_chunks": self.saved_chunks,
                "chunk_number": self.chunk_number,
                "model_id": self.model_id
            }
            block_full = self.current_block.save_cache(cache_state, seq_len)
            
            if block_full:
                logger.info(f"Block {self.current_block.block_id} is full ({self.current_block.block_used}/{self.block_size} tokens)")
                return True
        
        # TODO: handle the block full case directly in the add_knowlege function, instead of returning the flag?
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
        self.summary = self._agent_generate(instruction=summary_instruction, max_new_tokens=4096)

    def _agent_generate(self,max_new_tokens: int=1024, instruction: str=None, question: str=None)-> str:
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
        
        # Merge cache with distribution across devices
        merged_cache = DynamicCache()
        num_layers = len(self.saved_chunks[0]["cache"])
        for layer_idx in range(num_layers):
            keys, values = [], []
            target_device = self.available_devices[layer_idx % len(self.available_devices)]
            for chunk_info in self.saved_chunks:
                k, v = chunk_info["cache"][layer_idx]
                keys.append(k.to(target_device, non_blocking=True))
                values.append(v.to(target_device, non_blocking=True))
            merged_cache.update(torch.cat(keys, dim=2), torch.cat(values, dim=2), layer_idx)
        
        # Format query
        if question:
            # This means we are in the query mode
            formatted_query = f"\n\nBased on the context information provided above, please extract all the original information that is relevant to the question(**REMEBER to give EXACT datetime** along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'.):\n{question}{self.role_end}\n{self.role_start}assistant\n"
        elif instruction:
            formatted_query=f"\n\nBased on the context information provided above, please follow this instruction:\n{instruction}{self.role_end}\n{self.role_start}assistant\n"
        else:
            raise ValueError("Either question or instruction must be provided.")
        
        input_ids = self.tokenizer.encode(formatted_query, return_tensors="pt", add_special_tokens=False).to(self.primary_device)
        query_len = input_ids.shape[1]
        
        # Position IDs
        query_position_ids = torch.arange(self.global_offset, self.global_offset + query_len, dtype=torch.long, device=self.primary_device).unsqueeze(0)
        
        # Attention mask
        cache_length = merged_cache.get_seq_length()
        attention_mask = torch.ones((1, cache_length + query_len), dtype=torch.long, device=self.primary_device)
        
        # Manual generation with repetition penalty
        next_token_input = input_ids
        next_position_ids = query_position_ids
        past_kv = merged_cache
        current_position = self.global_offset + query_len
        
        eos_token_ids = [self.tokenizer.eos_token_id] if not isinstance(self.tokenizer.eos_token_id, (list, tuple)) else self.tokenizer.eos_token_id
        generated_tokens = []
        repetition_penalty = 1.1
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=next_token_input, attention_mask=attention_mask, position_ids=next_position_ids, past_key_values=past_kv, use_cache=True)
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
                next_position_ids = torch.tensor([[current_position]], dtype=torch.long, device=self.primary_device)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.primary_device)], dim=1)
                current_position += 1
                past_kv = outputs.past_key_values
                
                if next_token.item() in eos_token_ids:
                    break
        
        # Decode
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = self._remove_thinking_content(response)
        logger.debug(f"Generated response length: {len(response)} chars")
        
        return response
    
    def _remove_thinking_content(self, response: str) -> str:
        cleaned_response = response
        # Remove both <thinking> and <think> tags
        cleaned_response = re.sub(r'<thinking>.*?</thinking>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)  # 规范化多个空行
        cleaned_response = cleaned_response.strip()  # 去除首尾空白
        
        return cleaned_response
    
    def query(self, question: str, max_new_tokens: int = 1024) -> str:
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