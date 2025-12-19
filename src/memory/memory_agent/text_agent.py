import logging
import uuid
from datetime import datetime
from typing import List

from transformers import AutoTokenizer

from src.agent.base import BaseAgent
from src.memory.kv_block_manager.text_block import TextBlock
from src.utils.prompt import MEMORY_AGENT_SYS_PROMPT, SUMMARY_INSTRUCTION

logger = logging.getLogger(__name__)


class _TextMemoryLLM(BaseAgent):
    """Internal LLM wrapper for TextMemoryAgent."""
    def execute_tool(self, tool_name, arguments):
        # TextMemoryAgent doesn't use tools
        pass


class TextMemoryAgent:
    def __init__(self, model_id: str, openai_config: dict, model_context_window: int = 32768,
                 load_from_block_id: str = None, load_timestamp: str = None,
                 block_size_ratio:float=0.125):
        self.model_id = model_id
        self.model_context_window = model_context_window
        self.block_size_ratio=block_size_ratio
        self.block_size = int(model_context_window * block_size_ratio)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.llm = _TextMemoryLLM(openai_config, MEMORY_AGENT_SYS_PROMPT)
        self.is_active = True
        self.summary = None
        
        if load_from_block_id and load_timestamp:
            self.current_block = TextBlock(
                block_id=uuid.UUID(load_from_block_id),
                create_timestamp=load_timestamp,
                block_size=self.block_size
            )
            self.current_block.load()
            logger.info(f"Loaded existing text block: {load_from_block_id}")
        else:
            self.current_block = TextBlock(
                block_id=uuid.uuid4(),
                create_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                block_size=self.block_size
            )
            logger.info(f"TextMemoryAgent initialized with block_size={self.block_size}")

    def add(self, text_chunks: List[str]) -> bool:
        if not self.is_active:
            raise RuntimeError("Agent is inactive, cannot add new memories.")
        
        for text in text_chunks:
            token_count = len(self.tokenizer.encode(text))
            block_full = self.current_block.add_chunk(text, token_count)
            
            if block_full:
                logger.info(f"Block {self.current_block.block_id} is full")
                self.is_active = False
                self._create_summaries()
                return

    def _create_summaries(self):
        logger.info("Creating summary for text block")
        all_text = self.current_block.get_all_text()
        prompt = f"{all_text}\n\n{SUMMARY_INSTRUCTION}"
        self.summary = self.llm.generate_response(prompt, max_tokens=8192)
        logger.info(f"Summary created (length: {len(self.summary)} chars)")

    def preload_cache(self):
        """No-op for text storage mode (no cache to preload)."""
        pass

    def query(self, question: str, max_new_tokens: int = 8192) -> str:
        logger.debug(f"Querying text memory: {question[:50]}...")
        all_text = self.current_block.get_all_text()
        if not all_text:
            return "No knowledge available."
        
        prompt = f"{all_text}\n\nBased on the context information provided above, please extract the original information that is relevant to the question (REMEMBER to give EXACT datetime along with information, and the datetime format is 'YYYY-MM-DD HH:MM:SS'):\n{question}"
        return self.llm.generate_response(prompt, max_tokens=max_new_tokens)

    def get_original_texts(self) -> List[str]:
        """
        Get original text chunks for hybrid routing.
        
        Returns:
            List of original text chunks stored in this block.
        """
        return [chunk['text'] for chunk in self.current_block.chunks]

    @property
    def original_texts(self) -> List[str]:
        """Property alias for get_original_texts for compatibility."""
        return self.get_original_texts()