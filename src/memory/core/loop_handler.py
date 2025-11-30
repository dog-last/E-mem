from multiprocessing import Process
from typing import List

from memory.kv_block_manager.block import clear_cache
from memory.memory_agent.agent import MemoryAgent
from memory.router import Router


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

    def add_memory(self, text_chunks: List[str]) -> bool:
        """
        Add the text chunks to the active memory agent.
        Args:
            text_chunks (List[str]): The text chunks to add to the memory agent.
        Returns:
            bool: False if the memory agent can still be used to add more knowledge, True otherwise. Handle it in the outer handler
        """
        if self.active_memory_agent is None:
            self.create_agent()
        self.active_memory_agent.add(text_chunks)
        if self.active_memory_agent.is_active:
            return True
        return False
    
    def query_new_agent(self, text_chunks: List[str])->str:
        result=self.active_memory_agent.query(text_chunks)
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
        if router_system_prompt is None:
            self.query_handler=QueryHandler(Router(openai_config=openai_config))
        self.query_handler=QueryHandler(Router(openai_config=openai_config,system_prompt=router_system_prompt))
        if clean_cache_first:
            clear_cache()
        
    def add_memory(self,text_chunks: List[str]):
        """
        Add the text chunks to the active memory agent.
        Args:
            text_chunks (List[str]): The text chunks to add to the memory agent.
        """
        agent_full=self.add_handler.add_memory(text_chunks)
        if agent_full:
            self.inactive_memory_agent.append(self.add_handler.active_memory_agent)
            self.add_handler.active_memory_agent=None
            self.add_handler.create_agent()

    def query_memory(self,user_query:str)->str:
        """
        Query the memory agents for the user query.
        Args:
            user_query (str): The user query.
        Returns:
            str: The response from the memory agents.
        """
        old_memory_query_p=Process(target=self.query_handler.query_memory,args=(user_query,))
        new_memory_query_p=Process(target=self.add_handler.query_new_agent, args=(user_query, ))
        old_memory_query_p.start()
        new_memory_query_p.start()
        old_memory_query_p.join()
        new_memory_query_p.join()
        old_memory=old_memory_query_p.result
        new_memory=new_memory_query_p.result
        if old_memory=="No relevant memory found.":
            return new_memory
        all_memory=f"The memory stored a period of time ago: {old_memory}\n\nThe memory stored just now: {new_memory}"
        return all_memory