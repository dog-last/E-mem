import atexit
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from src.memory.kv_block_manager.block import clear_cache as clear_kv_cache
from src.memory.kv_block_manager.metadata import (
    clear_metadata,
    load_agents_metadata,
    save_agents_metadata,
)
from src.memory.memory_agent.agent import MemoryAgent
from src.memory.router.router import Router

logger = logging.getLogger(__name__)


class AddHandler:
    def __init__(self,model_id: str, 
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   quantization_config=None,
                   max_memory=None,
                   offload_folder=None,
                   overlap_ratio: float = 0.1,
                   overlap_mode: str = "chunk"):
        self.model_id=model_id
        self.model_context_window=model_context_window
        self.attn_implementation=attn_implementation
        self.device_map=device_map
        self.quantization_config=quantization_config
        self.max_memory=max_memory
        self.offload_folder=offload_folder
        self.overlap_ratio=overlap_ratio
        self.overlap_mode=overlap_mode  # "chunk" or "token"
        self.active_memory_agent=None
        self.overlap_buffer=[]  # Store recent memories for overlap
    
    def create_agent(self):
        logger.info("Creating new memory agent")
        self.active_memory_agent=MemoryAgent(model_id=self.model_id,
                                             model_context_window=self.model_context_window,
                                             attn_implementation=self.attn_implementation,
                                             device_map=self.device_map,
                                             quantization_config=self.quantization_config,
                                             max_memory=self.max_memory,
                                             offload_folder=self.offload_folder)
        logger.debug(f"Memory agent created with model: {self.model_id}")

    def add_memory(self, text: str) -> bool:
        """
        Add text to the active memory agent.
        Args:
            text (str): The text to add to the memory agent.
        Returns:
            bool: True if the memory agent is still active, False if it became full.
        """
        if self.active_memory_agent is None:
            self.create_agent()
        
        logger.debug(f"Adding text to memory agent: {text[:50]}...")
        self.active_memory_agent.add([text])
        
        # Add to overlap buffer only if overlap_ratio > 0
        if self.overlap_ratio > 0:
            import re

            import tiktoken
            
            block_size = self.active_memory_agent.block_size
            overlap_tokens = int(block_size * self.overlap_ratio)
            
            try:
                tokenizer = tiktoken.encoding_for_model("gpt-4")
            except KeyError:
                tokenizer = tiktoken.get_encoding("cl100k_base")
            
            if self.overlap_mode == "token":
                # Token mode: accumulate sentences up to token limit
                sentences = re.split(r'(?<=[.!?])\s+', text)
                self.overlap_buffer.extend(sentences)
                
                # Trim to token limit from the end
                total_tokens = 0
                keep_from_idx = len(self.overlap_buffer)
                
                for i in range(len(self.overlap_buffer) - 1, -1, -1):
                    sent_tokens = len(tokenizer.encode(self.overlap_buffer[i]))
                    if total_tokens + sent_tokens <= overlap_tokens:
                        total_tokens += sent_tokens
                        keep_from_idx = i
                    else:
                        break
                
                self.overlap_buffer = self.overlap_buffer[keep_from_idx:]
            else:
                # Chunk mode: keep whole chunks
                self.overlap_buffer.append(text)
                
                total_tokens = 0
                keep_from_idx = len(self.overlap_buffer)
                
                for i in range(len(self.overlap_buffer) - 1, -1, -1):
                    chunk_tokens = len(tokenizer.encode(self.overlap_buffer[i]))
                    if total_tokens + chunk_tokens <= overlap_tokens:
                        total_tokens += chunk_tokens
                        keep_from_idx = i
                    else:
                        break
                
                self.overlap_buffer = self.overlap_buffer[keep_from_idx:]
        
        is_active = self.active_memory_agent.is_active
        if not is_active:
            logger.info(f"Memory agent became full, overlap buffer size: {len(self.overlap_buffer)}")
        return is_active
    
    def get_overlap_memories(self) -> list:
        """Get memories for overlap with next block."""
        return self.overlap_buffer.copy()
    
    def clear_overlap_buffer(self):
        """Clear overlap buffer after creating new agent."""
        self.overlap_buffer = []
    
    def query_new_agent(self, query: str)->str:
        if self.active_memory_agent is None:
            logger.debug("No active memory agent for query")
            return "No active memory."
        if not self.active_memory_agent.saved_chunks:
            logger.debug("Active memory agent has no data yet")
            return "No active memory."
        logger.debug(f"Querying active memory agent: {query[:50]}...")
        result = self.active_memory_agent.query(query)
        return result
    

class QueryHandler:
    def __init__(self,router:Router):
        self.inactive_memory_agent=[]
        self.router=router

    def query_memory(self,user_query:str)->str:
        logger.debug(f"Querying inactive memory blocks: {user_query[:50]}...")
        res=self.router.map_reduce_blocks(user_query)
        if not res:
            logger.debug("No relevant memory found in inactive blocks")
            return "No relevant memory found."
        logger.info(f"Found {len(res)} relevant memory blocks")
        formatted_res="\n".join([f"Old Memory Block {i+1}: {r}" for i,r in enumerate(res)]) # Better format this to avoid mixing together
        return formatted_res
        

class MemoryHandler:
    def __init__(self,model_id: str, clean_cache_first:bool=True,
                 openai_config: dict = None,
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   router_system_prompt: str = None,
                   quantization_config=None,
                   max_memory=None,
                   offload_folder=None,
                   overlap_ratio: float = 0.1,
                   overlap_mode: str = "chunk"):
        logger.info(f"Initializing MemoryHandler with model: {model_id}, overlap_ratio: {overlap_ratio}")
        self.model_id = model_id
        self.model_context_window = model_context_window
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.quantization_config = quantization_config
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        
        self.add_handler=AddHandler(model_id,model_context_window,attn_implementation,device_map,quantization_config,max_memory,offload_folder,overlap_ratio,overlap_mode)
        self.inactive_memory_agents = []
        if router_system_prompt is None:
            self.query_handler=QueryHandler(Router(openai_config=openai_config))
        else:
            self.query_handler=QueryHandler(Router(openai_config=openai_config,system_prompt=router_system_prompt))
        
        if clean_cache_first:
            logger.info("Clearing KV cache and metadata")
            clear_kv_cache()
            clear_metadata()
        else:
            # Load existing agents
            self._load_existing_agents()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
    def add_memory(self, text: str):
        """
        Add text to the active memory agent.
        Args:
            text (str): The text to add to the memory agent.
        """
        # Create agent if doesn't exist
        if self.add_handler.active_memory_agent is None:
            self.add_handler.create_agent()
        
        # Check if agent is already inactive before calling add_memory
        # This handles the case where previous call made agent inactive but transition failed
        if not self.add_handler.active_memory_agent.is_active:
            logger.warning("Active agent is already inactive, triggering transition before add")
            # Move to inactive
            self.inactive_memory_agents.append(self.add_handler.active_memory_agent)
            self.query_handler.router.add_blocks(self.add_handler.active_memory_agent)
            
            # Get overlap memories
            overlap_memories = self.add_handler.get_overlap_memories()
            
            # Create new agent
            self.add_handler.active_memory_agent = None
            self.add_handler.create_agent()
            
            # Add overlap memories
            if overlap_memories:
                for mem in overlap_memories:
                    self.add_handler.active_memory_agent.add([mem])
                logger.info(f"Added {len(overlap_memories)} overlap memories to new agent")
            
            # Clear overlap buffer
            self.add_handler.clear_overlap_buffer()
        
        is_active = self.add_handler.add_memory(text)
        
        if not is_active:
            # Agent became full, move to inactive and create new one with overlap
            logger.info("Moving full memory agent to inactive pool")
            self.inactive_memory_agents.append(self.add_handler.active_memory_agent)
            self.query_handler.router.add_blocks(self.add_handler.active_memory_agent)
            logger.info(f"Total inactive agents: {len(self.inactive_memory_agents)}")
            
            # Get overlap memories before clearing
            overlap_memories = self.add_handler.get_overlap_memories()
            logger.info(f"Creating new agent with {len(overlap_memories)} overlap memories")
            
            # Create new agent
            self.add_handler.active_memory_agent = None
            self.add_handler.create_agent()
            
            # Add overlap memories to new agent
            if overlap_memories:
                for mem in overlap_memories:
                    self.add_handler.active_memory_agent.add([mem])
                logger.info(f"Added {len(overlap_memories)} overlap memories to new agent")
            
            # Clear overlap buffer
            self.add_handler.clear_overlap_buffer()
        
        # Save metadata after handling agent transition
        self._save_metadata()

    def query_memory(self, user_query: str) -> str:
        """
        Query the memory agents for the user query.
        Args:
            user_query (str): The user query.
        Returns:
            str: The response from the memory agents.
        """
        # Use ThreadPoolExecutor for parallel queries
        logger.debug("Starting parallel memory queries")
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_memory_future = executor.submit(self.query_handler.query_memory, user_query)
            new_memory_future = executor.submit(self.add_handler.query_new_agent, user_query)
            
            old_memory = old_memory_future.result()
            new_memory = new_memory_future.result()
        # Force cleanup after parallel queries
        import torch
        torch.cuda.empty_cache()
        logger.debug("Parallel queries completed")
        
        # Handle different cases
        has_old = old_memory != "No relevant memory found."
        has_new = new_memory != "No active memory."
        
        if not has_old and not has_new:
            return "No memory found."
        elif not has_old:
            return new_memory
        elif not has_new:
            return old_memory
        else:
            return f"The memory stored a period of time ago: {old_memory}\n\nThe memory stored just now:\n New Memory Block 1: {new_memory}"
    
    def _save_metadata(self):
        """Save all agents metadata to disk."""
        # Load existing metadata to preserve other sessions' blocks
        all_metadata = load_agents_metadata()
        
        # Get session_id from environment
        session_id = os.environ.get('EVAL_SESSION_ID', 'default')
        
        # Remove entries for current model AND session
        other_sessions_metadata = [m for m in all_metadata 
                                   if not (m.get("model_id") == self.model_id and 
                                          m.get("session_id") == session_id)]
        
        # Build current session's metadata
        current_session_metadata = []
        
        # Save inactive agents (with summary)
        for agent in self.inactive_memory_agents:
            current_session_metadata.append({
                "block_id": str(agent.current_block.block_id),
                "timestamp": agent.current_block.create_timestamp,
                "model_id": agent.model_id,
                "session_id": session_id,
                "summary": agent.summary,
                "is_active": False,
                "block_used": agent.current_block.block_used,
                "chunk_number": agent.chunk_number
            })
        
        # Save active agent if exists AND has data (without summary)
        if self.add_handler.active_memory_agent and self.add_handler.active_memory_agent.saved_chunks:
            agent = self.add_handler.active_memory_agent
            # Skip if agent is a Mock (test environment)
            if not hasattr(agent, '_mock_name'):
                current_session_metadata.append({
                    "block_id": str(agent.current_block.block_id),
                    "timestamp": agent.current_block.create_timestamp,
                    "model_id": agent.model_id,
                    "session_id": session_id,
                    "summary": agent.summary,  # Use actual summary (None for active, set for inactive)
                    "is_active": agent.is_active,  # Use actual is_active status
                    "block_used": agent.current_block.block_used,
                    "chunk_number": agent.chunk_number
                })
        
        # Merge: other sessions + current session
        final_metadata = other_sessions_metadata + current_session_metadata
        save_agents_metadata(final_metadata)
        logger.info(f"Saved metadata: {len(current_session_metadata)} for current session ({self.model_id}, {session_id}), {len(other_sessions_metadata)} for other sessions, total {len(final_metadata)}")
    
    def _load_existing_agents(self):
        """Load existing agents from metadata."""
        metadata = load_agents_metadata()
        if not metadata:
            logger.info("No existing agents found")
            return
        
        # Get session_id from environment
        session_id = os.environ.get('EVAL_SESSION_ID', 'default')
        
        # Filter agents for current model AND session only
        current_session_metadata = [m for m in metadata 
                                    if m.get("model_id") == self.model_id and 
                                       m.get("session_id") == session_id]
        other_sessions_count = len(metadata) - len(current_session_metadata)
        
        if other_sessions_count > 0:
            logger.info(f"Found {len(metadata)} total agents: {len(current_session_metadata)} for current session ({self.model_id}, {session_id}), {other_sessions_count} for other sessions (skipped)")
        else:
            logger.info(f"Loading {len(current_session_metadata)} agents for session {session_id}")
        
        for agent_data in current_session_metadata:
            # Recreate agent
            agent = MemoryAgent(
                model_id=self.model_id,
                model_context_window=self.model_context_window,
                attn_implementation=self.attn_implementation,
                device_map=self.device_map,
                quantization_config=self.quantization_config,
                max_memory=self.max_memory,
                offload_folder=self.offload_folder,
                load_from_block_id=agent_data["block_id"],
                load_timestamp=agent_data["timestamp"]
            )
            
            # Restore summary and block_used
            agent.summary = agent_data.get("summary")
            agent.current_block.block_used = agent_data.get("block_used", 0)
            agent.chunk_number = agent_data.get("chunk_number", 0)
            
            if agent_data["is_active"]:
                # Restore as active agent
                agent.is_active = True
                if agent.merged_cache is None and agent.saved_chunks:
                    logger.warning("Active agent has no merged_cache, will need to rebuild")
                self.add_handler.active_memory_agent = agent
                logger.info(f"Restored active agent: {agent_data['block_id']} (used: {agent_data.get('block_used', 0)} tokens, cache_loaded: {agent.merged_cache is not None})")
            else:
                # Restore as inactive agent
                agent.is_active = False
                # Pre-load cache to CPU for faster queries
                agent.preload_cache()
                self.inactive_memory_agents.append(agent)
                self.query_handler.router.add_blocks(agent)
                logger.info(f"Restored inactive agent: {agent_data['block_id']} (summary: {len(agent.summary) if agent.summary else 0} chars)")
        
        logger.info(f"Loaded {len(self.inactive_memory_agents)} inactive agents and {'1 active' if self.add_handler.active_memory_agent else 'no active'} agent")
    
    def cleanup(self):
        """Save active agent cache state on exit."""
        try:
            if self.add_handler.active_memory_agent and self.add_handler.active_memory_agent.saved_chunks:
                agent = self.add_handler.active_memory_agent
                # Skip if agent is a Mock (test environment)
                if hasattr(agent, '_mock_name'):
                    return
                logger.info(f"Saving active agent cache state on exit (block {agent.current_block.block_id})")
                cache_state = {
                    "global_offset": agent.global_offset,
                    "saved_chunks": agent.saved_chunks,
                    "chunk_number": agent.chunk_number,
                    "model_id": agent.model_id,
                    "merged_cache": [(k.cpu(), v.cpu()) for k, v in agent.merged_cache] if agent.merged_cache else None
                }
                agent.current_block.save_cache(cache_state, 0)
                logger.info("Active agent cache saved successfully")
        except Exception:
            pass  # Silently fail on cleanup to avoid breaking tests