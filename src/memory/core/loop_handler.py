import logging
from concurrent.futures import ThreadPoolExecutor

from src.memory.kv_block_manager.block import clear_cache
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
                   overlap_ratio: float = 0.1):
        self.model_id=model_id
        self.model_context_window=model_context_window
        self.attn_implementation=attn_implementation
        self.device_map=device_map
        self.quantization_config=quantization_config
        self.max_memory=max_memory
        self.offload_folder=offload_folder
        self.overlap_ratio=overlap_ratio
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
        
        # Add to overlap buffer
        self.overlap_buffer.append(text)
        
        # Calculate overlap size based on block size
        overlap_size = int(self.active_memory_agent.block_size * self.overlap_ratio)
        
        # Keep only recent memories for overlap (estimate ~100 tokens per memory)
        max_buffer_items = max(5, overlap_size // 100)
        if len(self.overlap_buffer) > max_buffer_items:
            self.overlap_buffer = self.overlap_buffer[-max_buffer_items:]
        
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
                   overlap_ratio: float = 0.1):
        logger.info(f"Initializing MemoryHandler with model: {model_id}, overlap_ratio: {overlap_ratio}")
        self.add_handler=AddHandler(model_id,model_context_window,attn_implementation,device_map,quantization_config,max_memory,offload_folder,overlap_ratio)
        self.inactive_memory_agents = []
        if router_system_prompt is None:
            self.query_handler=QueryHandler(Router(openai_config=openai_config))
        else:
            self.query_handler=QueryHandler(Router(openai_config=openai_config,system_prompt=router_system_prompt))
        if clean_cache_first:
            logger.info("Clearing KV cache")
            clear_cache()
        
    def add_memory(self, text: str):
        """
        Add text to the active memory agent.
        Args:
            text (str): The text to add to the memory agent.
        """
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