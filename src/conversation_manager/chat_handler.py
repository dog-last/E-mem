import logging

from src.agent.base import BaseAgent
from src.memory.core.loop_handler import MemoryHandler
from src.utils.prompt import CHAT_SYS_PROMPT

logger = logging.getLogger(__name__)


class ChatManager(BaseAgent):
    def __init__(self,model_id:str,openai_config:dict=None,
                 system_prompt:str=CHAT_SYS_PROMPT,
                 clean_cache_first:bool=True,
                 model_context_window: int =32768, 
                 attn_implementation: str = "sdpa",
                   device_map: str = "auto", 
                   router_system_prompt: str = None,
                   quantization_config=None,
                   max_memory=None,
                   offload_folder=None):
        super().__init__(openai_config,system_prompt)
        self.name="chat_manager"
        self.last_queried_memory = None  # Store last query result
        logger.info(f"Initializing ChatManager with model: {model_id}")
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
        if max_memory is not None:
            memory_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            memory_kwargs["offload_folder"] = offload_folder
        
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
            auto_save (bool): Whether to auto save the memory. It will auto save the user_input directly without LLM processing.
            save_original_input (bool): Whether to save the original input. It will overwrite the parameter in the tool calling.
            max_new_tokens (int): The max new tokens.
        Returns:
            str: The chat response.
        """
        self.handle_user_input=user_input
        self.auto_save=auto_save
        self.save_original_input=save_original_input
        
        # Auto-save mode: directly save without LLM processing
        if auto_save:
            logger.debug("Auto-save mode: directly saving input")
            target_memory = user_input
            result = self.add_memory(target_memory)
            return result
        
        # Normal mode: let LLM decide
        tools=[] if outer_tools is None else outer_tools.copy()
        tools.append(self.add_mem_tool)        
        tools.append(self.search_mem_tool)
        
        user_prompt_formatted = f"""Please read the user input carefully and answer the question or follow the instructions to finish the tasks:
        <user_input>{user_input}</user_input>
        """
        
        try:
            response=self.generate_response(user_prompt_formatted,tools=tools,max_tokens=max_new_tokens,max_tool_rounds=1)
        except RuntimeError as e:
            logger.error(f"RuntimeError in generate_response: {e}", exc_info=True)
            return f"Not mentioned in the conversation."
        return response


    def execute_tool(self, tool_name, arguments):
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")
        if tool_name == "add_memory":
            return self.add_memory(arguments.get("memory"))
        elif tool_name == "query_memory":
            return self.search_memory(arguments.get("query"))
        else:
            logger.error(f"Unknown tool: {tool_name}")
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
            logger.warning("No memory content provided")
            return "[ERROR] No memory content provided."
        try:
            logger.info(f"Adding memory: {target_memory[:100]}...")
            self.memory_handler.add_memory(target_memory)
            logger.info("Memory added successfully")
        except Exception as e:
            logger.error(f"Memory adding failed: {e}", exc_info=True)
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
            logger.warning("No query content provided")
            return "[ERROR] No query content provided."
        try:
            logger.info(f"Querying memory: {query}")
            result=self.memory_handler.query_memory(query)
            logger.info(f"Memory query completed, result: {result}")
            # Store the queried memory for evaluation
            self.last_queried_memory = result
        except Exception as e:
            logger.error(f"Memory querying failed: {e}", exc_info=True)
            self.last_queried_memory = None
            return f"[ERROR] Memory querying failed: {e}"
        return result
        