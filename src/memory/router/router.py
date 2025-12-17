import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from src.agent.base import BaseAgent
from src.utils.parser import limit_memory_segments
from src.utils.prompt import ROUTER_SYS_PROMPT

logger = logging.getLogger(__name__)


class Router(BaseAgent):
    def __init__(
        self,
        openai_config: dict = None,
        system_prompt: str = ROUTER_SYS_PROMPT,
        max_memory_segments: Optional[int] = None,
        max_blocks: int = 5,
    ) -> None:
        if not openai_config:
            raise NotImplementedError("Please provide openai_config for router.")
        super().__init__(openai_config, system_prompt)
        self.name = "router"
        self.agent = []
        self.max_memory_segments = max_memory_segments
        self.max_blocks = max_blocks
        logger.info(f"Router initialized (max_memory_segments={max_memory_segments}, max_blocks={max_blocks})")

    def add_blocks(self, memory_agent):
        """Add inactive memory agent to router for querying."""
        if memory_agent.is_active:
            logger.warning("Attempted to add active memory agent to router")
            return
        self.agent.append(memory_agent)
        logger.info(f"Added memory block to router, total blocks: {len(self.agent)}")

    def _map_blocks(self, user_query: str, max_blocks: Optional[int] = None) -> list:
        """
        Map the user query to relevant memory agents.

        Args:
            user_query (str): The user query.
            max_blocks (int): The maximum number of blocks to map. Uses self.max_blocks if None.

        Returns:
            list: A list of memory agents.
        """
        if max_blocks is None:
            max_blocks = self.max_blocks

        if not self.agent:
            logger.debug("No memory agents available for mapping")
            return []

        summary_blocks = "\n".join(
            map(
                lambda idx_agent: f"""
            <summary>
                <index>{idx_agent[0]}</index>
                <content>{idx_agent[1].summary}</content>
            </summary>
            """,
                enumerate(self.agent),
            )
        )

        user_prompt_formatted = f"""
            <query> {user_query} </query>
            <summary_list>
            {summary_blocks}
            </summary_list>
            Please provide the indices of the most relevant memory summaries to the query.
        """
        response = self.generate_response(user_prompt_formatted, max_tokens=4096)

        # Extract indices using robust regex
        try:
            # Match <summary_index>...</summary_index> - capture anything inside
            match = re.search(
                r"<summary_index>\s*(.*?)\s*</summary_index>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                indices_str = match.group(1)
                # Extract all integers
                indices = [int(num) for num in re.findall(r"\d+", indices_str)]

                # Deduplicate while preserving order
                seen = set()
                unique_indices = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)

                # Filter valid indices first, then limit to max_blocks
                valid_indices = [
                    idx for idx in unique_indices if 0 <= idx < len(self.agent)
                ]
                selected_indices = valid_indices[:max_blocks]
                selected_agents = [self.agent[idx] for idx in selected_indices]

                logger.info(
                    f"Mapped query to {len(selected_agents)} memory blocks (indices: {selected_indices})"
                )
                return selected_agents
            else:
                logger.warning(
                    f"No summary_index tag found, using all blocks. Response: {response[:200]}"
                )
                return self.agent[:max_blocks]
        except Exception as e:
            logger.error(
                f"Error parsing router response: {e}, using all blocks", exc_info=True
            )
            return self.agent[:max_blocks]
        
    
    def execute_tool(self, tool_name, arguments):
        pass


    def map_reduce_blocks(self, user_query: str) -> List[str]:
        """
        Collect all the responses from the relevant memory agents.

        Args:
            user_query (str): The user query.

        Returns:
            list: A list of responses.
        """
        relevant_agents_list = self._map_blocks(user_query)
        if not relevant_agents_list:
            logger.debug("No relevant agents found for query")
            return []

        # Pre-load all caches in parallel (disk I/O only, no GPU)
        logger.debug(
            f"Pre-loading {len(relevant_agents_list)} caches from disk in parallel"
        )
        with ThreadPoolExecutor(max_workers=len(relevant_agents_list)) as executor:
            list(executor.map(lambda agent: agent.preload_cache(), relevant_agents_list))

        # Now query in parallel (GPU operations limited by semaphore)
        logger.debug(
            f"Querying {len(relevant_agents_list)} relevant memory blocks in parallel"
        )
        with ThreadPoolExecutor(max_workers=len(relevant_agents_list)) as executor:
            results = list(
                executor.map(lambda agent: agent.query(user_query), relevant_agents_list)
            )

        # Apply memory segment limit if configured
        if self.max_memory_segments is not None and self.max_memory_segments > 0:
            logger.debug(
                f"Applying memory segment limit: max_memory_segments={self.max_memory_segments}"
            )
            results = [
                limit_memory_segments(result, self.max_memory_segments)
                for result in results
            ]

        # Force cleanup after parallel queries
        import torch

        torch.cuda.empty_cache()
        logger.info(f"Collected {len(results)} results from memory blocks")
        return results