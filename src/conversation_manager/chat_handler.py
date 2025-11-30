from src.agent.base import BaseAgent
from src.memory.core.loop_handler import MemoryHandler
from src.utils.prompt import CHAT_SYS_PROMPT


class ChatManager(BaseAgent):
    def __init__(self,model_id:str,openai_config:dict=None,
                 system_prompt:str=CHAT_SYS_PROMPT,
                 clean_cache_first:bool=True,
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   router_system_prompt: str = None,
                   quantization_config=None):
        super().__init__(openai_config,system_prompt)
        self.name="chat_manager"
        self.add_mem_tool={
                "type":"function",
                "function":{
                    "name":"add_memory",
                    "description":"This can store some information into memory blocks, so that you can use it in the future.",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "memory":{
                                "type":"string",
                                "description":"The memory content to be stored."
                            }
                        },
                        "required":["memory"]
                    }
                }
            }
        
        self.search_mem_tool={
                "type":"function",
                "function":{
                    "name":"query_memory",
                    "description":"This can query some information from memory blocks, so that you can use it to answer user questions.",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "query":{
                                "type":"string",
                                "description":"The query content to be used to query memory."
                            }
                        },
                        "required":["query"]
                    }
                }
            }
        self.auto_save=False
        self.save_original_input=False
        self.handle_user_input=None

        # Build kwargs, excluding None values to preserve defaults
        memory_kwargs = {
            "model_id": model_id,
            "openai_config": openai_config,
            "clean_cache_first": clean_cache_first,
            "model_context_window": model_context_window,
            "attn_implementation": attn_implementation,
            "device_map": device_map
        }
        if router_system_prompt is not None:
            memory_kwargs["router_system_prompt"] = router_system_prompt
        if quantization_config is not None:
            memory_kwargs["quantization_config"] = quantization_config
        
        self.memory_handler = MemoryHandler(**memory_kwargs)
        
    def chat(self,user_input:str,outer_tools=None,
             auto_save:bool=False,
             save_original_input:bool=False,
             max_new_tokens:int=1024)->str:
        """
        Chat with the user.
        Args:
            user_input (str): The user input.
            outer_tools (list): The outer tools.
            auto_save (bool): Whether to auto save the memory. It will auto save the user_input, and will not ask agent to decide.
            save_original_input (bool): Whether to save the original input. It will overwrite the parameter in the tool calling.
            max_new_tokens (int): The max new tokens.
        Returns:
            str: The chat response.
        """
        self.handle_user_input=user_input
        tools=[] if outer_tools is None else outer_tools.copy()
        
        tools.append(self.add_mem_tool)        
        tools.append(self.search_mem_tool)
        
        self.auto_save=auto_save
        self.save_original_input=save_original_input
        
        user_prompt_formatted = f"""Please read the user input carefully and answer the question or follow the instructions to finish the tasks:
        <user_input>{user_input}</user_input>
        """
        
        response=self.generate_response(user_prompt_formatted,tools=tools,max_tokens=max_new_tokens,max_tool_rounds=1)
        return response


    def execute_tool(self, tool_name, arguments):
        print(f"Executing tool: {tool_name} with arguments: {arguments}")
        if tool_name == "add_memory":
            return self.add_memory(arguments.get("memory"))
        elif tool_name == "query_memory":
            return self.search_memory(arguments.get("query"))
        else:
            return f"[ERROR] Unknown tool: {tool_name}"

    
    def add_memory(self, memory: str) -> str:
        """
        Add memory to the memory blocks.
        Args:
            memory (str): The memory content to be stored.
        Returns:
            str: Success or failure message.
        """
        target_memory = self.handle_user_input if self.save_original_input else memory
        if not target_memory:
            return "[ERROR] No memory content provided."
        try:
            self.memory_handler.add_memory(target_memory)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[ERROR] Memory adding failed: {e}"
        return "[SUCCESS] Memory added successfully."
    
    def search_memory(self,query:str)->str:
        """
        Search memory blocks with the query.
        Args:
            query (str): The query content to be used to query memory.
        Returns:
            str: The search result.
        """
        if not query:
            return "[ERROR] No query content provided."
        try:
            result=self.memory_handler.query_memory(query)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[ERROR] Memory querying failed: {e}"
        return result
        