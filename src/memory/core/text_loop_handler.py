import logging
from concurrent.futures import ThreadPoolExecutor

from src.memory.kv_block_manager.text_block import (
    clear_text_cache,
    clear_text_metadata,
    load_text_agents_metadata,
    save_text_agents_metadata,
)
from src.memory.memory_agent.text_agent import TextMemoryAgent
from src.memory.router.router import Router

logger = logging.getLogger(__name__)


class TextAddHandler:
    def __init__(self, model_id: str, openai_config: dict, model_context_window: int = 32768, overlap_ratio: float = 0.1, overlap_mode: str = "chunk"):
        self.model_id = model_id
        self.openai_config = openai_config
        self.model_context_window = model_context_window
        self.overlap_ratio = overlap_ratio
        self.overlap_mode = overlap_mode  # "chunk" or "token"
        self.active_memory_agent = None
        self.overlap_buffer = []

    def create_agent(self):
        logger.info("Creating new text memory agent")
        self.active_memory_agent = TextMemoryAgent(
            model_id=self.model_id,
            openai_config=self.openai_config,
            model_context_window=self.model_context_window
        )

    def add_memory(self, text: str) -> bool:
        if self.active_memory_agent is None:
            self.create_agent()
        
        self.active_memory_agent.add([text])
        
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
        
        return self.active_memory_agent.is_active

    def get_overlap_memories(self) -> list:
        return self.overlap_buffer.copy()

    def clear_overlap_buffer(self):
        self.overlap_buffer = []

    def query_new_agent(self, query: str) -> str:
        if self.active_memory_agent is None:
            return "No active memory."
        return self.active_memory_agent.query(query)


class TextQueryHandler:
    def __init__(self, router: Router):
        self.inactive_memory_agent = []
        self.router = router

    def query_memory(self, user_query: str) -> str:
        res = self.router.map_reduce_blocks(user_query)
        if not res:
            return "No relevant memory found."
        return "\n".join([f"Old Memory Block {i+1}: {r}" for i, r in enumerate(res)])


class TextMemoryHandler:
    def __init__(self, model_id: str, openai_config: dict, clean_cache_first: bool = True,
                 model_context_window: int = 32768, router_system_prompt: str = None, overlap_ratio: float = 0.1, overlap_mode: str = "chunk"):
        logger.info(f"Initializing TextMemoryHandler with model: {model_id}")
        self.model_id = model_id
        self.openai_config = openai_config
        self.model_context_window = model_context_window
        self.add_handler = TextAddHandler(model_id, openai_config, model_context_window, overlap_ratio, overlap_mode)
        self.inactive_memory_agents = []
        
        if router_system_prompt is None:
            self.query_handler = TextQueryHandler(Router(openai_config=openai_config))
        else:
            self.query_handler = TextQueryHandler(Router(openai_config=openai_config, system_prompt=router_system_prompt))
        
        if clean_cache_first:
            logger.info("Clearing text cache and metadata")
            clear_text_cache()
            clear_text_metadata()
        else:
            self._load_existing_agents()

    def add_memory(self, text: str):
        if self.add_handler.active_memory_agent is None:
            self.add_handler.create_agent()
            self._save_metadata()
        
        # Check if agent is already inactive before calling add_memory
        if not self.add_handler.active_memory_agent.is_active:
            logger.warning("Active agent is already inactive, triggering transition before add")
            self.inactive_memory_agents.append(self.add_handler.active_memory_agent)
            self.query_handler.router.add_blocks(self.add_handler.active_memory_agent)
            
            overlap_memories = self.add_handler.get_overlap_memories()
            
            self.add_handler.active_memory_agent = None
            self.add_handler.create_agent()
            
            if overlap_memories:
                for mem in overlap_memories:
                    self.add_handler.active_memory_agent.add([mem])
                logger.info(f"Added {len(overlap_memories)} overlap memories to new agent")
            
            self.add_handler.clear_overlap_buffer()
        
        is_active = self.add_handler.add_memory(text)
        if not is_active:
            logger.info("Moving full text memory agent to inactive pool")
            self.inactive_memory_agents.append(self.add_handler.active_memory_agent)
            self.query_handler.router.add_blocks(self.add_handler.active_memory_agent)
            
            overlap_memories = self.add_handler.get_overlap_memories()
            logger.info(f"Creating new agent with {len(overlap_memories)} overlap memories")
            
            self.add_handler.active_memory_agent = None
            self.add_handler.create_agent()
            
            if overlap_memories:
                for mem in overlap_memories:
                    self.add_handler.active_memory_agent.add([mem])
            
            self.add_handler.clear_overlap_buffer()
            self._save_metadata()

    def query_memory(self, user_query: str) -> str:
        logger.debug("Starting parallel memory queries")
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_memory_future = executor.submit(self.query_handler.query_memory, user_query)
            new_memory_future = executor.submit(self.add_handler.query_new_agent, user_query)
            
            old_memory = old_memory_future.result()
            new_memory = new_memory_future.result()
        
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
        agents_data = []
        for agent in self.inactive_memory_agents:
            agents_data.append({
                "block_id": str(agent.current_block.block_id),
                "timestamp": agent.current_block.create_timestamp,
                "summary": agent.summary,
                "is_active": False,
                "block_used": agent.current_block.block_used,
                "chunk_number": agent.current_block.chunk_num
            })
        if self.add_handler.active_memory_agent:
            agent = self.add_handler.active_memory_agent
            agents_data.append({
                "block_id": str(agent.current_block.block_id),
                "timestamp": agent.current_block.create_timestamp,
                "summary": None,
                "is_active": True,
                "block_used": agent.current_block.block_used,
                "chunk_number": agent.current_block.chunk_num
            })
        save_text_agents_metadata(agents_data)
        logger.info(f"Saved metadata for {len(agents_data)} text agents")
    
    def _load_existing_agents(self):
        metadata = load_text_agents_metadata()
        if not metadata:
            logger.info("No existing text agents found")
            return
        logger.info(f"Loading {len(metadata)} existing text agents")
        for agent_data in metadata:
            agent = TextMemoryAgent(
                model_id=self.model_id,
                openai_config=self.openai_config,
                model_context_window=self.model_context_window,
                load_from_block_id=agent_data["block_id"],
                load_timestamp=agent_data["timestamp"]
            )
            agent.summary = agent_data.get("summary")
            agent.current_block.block_used = agent_data.get("block_used", 0)
            agent.current_block.chunk_num = agent_data.get("chunk_number", 0)
            if agent_data["is_active"]:
                agent.is_active = True
                self.add_handler.active_memory_agent = agent
                logger.info(f"Restored active text agent: {agent_data['block_id']}")
            else:
                agent.is_active = False
                self.inactive_memory_agents.append(agent)
                self.query_handler.router.add_blocks(agent)
                logger.info(f"Restored inactive text agent: {agent_data['block_id']}")
        logger.info(f"Loaded {len(self.inactive_memory_agents)} inactive agents and {'1 active' if self.add_handler.active_memory_agent else 'no active'} agent")
