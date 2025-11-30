from concurrent.futures import ThreadPoolExecutor

from src.memory.kv_block_manager.block import clear_cache
from src.memory.memory_agent.agent import MemoryAgent
from src.memory.router.router import Router


class AddHandler:
    def __init__(self,model_id: str, 
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   quantization_config=None):
        self.model_id=model_id
        self.model_context_window=model_context_window
        self.attn_implementation=attn_implementation
        self.device_map=device_map
        self.quantization_config=quantization_config
        self.active_memory_agent=None
    
    def create_agent(self):
        self.active_memory_agent=MemoryAgent(model_id=self.model_id,
                                             model_context_window=self.model_context_window,
                                             attn_implementation=self.attn_implementation,
                                             device_map=self.device_map,
                                             quantization_config=self.quantization_config)

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
        self.active_memory_agent.add([text])
        return self.active_memory_agent.is_active
    
    def query_new_agent(self, query: str)->str:
        if self.active_memory_agent is None:
            return "No active memory."
        result = self.active_memory_agent.query(query)
        return result
    

class QueryHandler:
    def __init__(self,router:Router):
        self.inactive_memory_agent=[]
        self.router=router

    def query_memory(self,user_query:str)->str:
        res=self.router.map_reduce_blocks(user_query)
        if not res:
            return "No relevant memory found."
        return "\n".join(res)
        

class MemoryHandler:
    def __init__(self,model_id: str, clean_cache_first:bool=True,
                 openai_config: dict = None,
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   router_system_prompt: str = None,
                   quantization_config=None):
        self.add_handler=AddHandler(model_id,model_context_window,attn_implementation,device_map,quantization_config)
        self.inactive_memory_agents = []
        if router_system_prompt is None:
            self.query_handler=QueryHandler(Router(openai_config=openai_config))
        else:
            self.query_handler=QueryHandler(Router(openai_config=openai_config,system_prompt=router_system_prompt))
        if clean_cache_first:
            clear_cache()
        
    def add_memory(self, text: str):
        """
        Add text to the active memory agent.
        Args:
            text (str): The text to add to the memory agent.
        """
        is_active = self.add_handler.add_memory(text)
        if not is_active:
            # Agent became full, move to inactive and create new one
            self.inactive_memory_agents.append(self.add_handler.active_memory_agent)
            self.query_handler.router.add_blocks(self.add_handler.active_memory_agent)
            self.add_handler.active_memory_agent = None
            self.add_handler.create_agent()

    def query_memory(self, user_query: str) -> str:
        """
        Query the memory agents for the user query.
        Args:
            user_query (str): The user query.
        Returns:
            str: The response from the memory agents.
        """
        # Use ThreadPoolExecutor for parallel queries
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_memory_future = executor.submit(self.query_handler.query_memory, user_query)
            new_memory_future = executor.submit(self.add_handler.query_new_agent, user_query)
            
            old_memory = old_memory_future.result()
            new_memory = new_memory_future.result()
        
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
            return f"The memory stored a period of time ago: {old_memory}\n\nThe memory stored just now: {new_memory}"