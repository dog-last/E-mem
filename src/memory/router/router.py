from agent.base import BaseAgent
from utils.prompt import ROUTER_SYS_PROMPT


class Router(BaseAgent):
    def __init__(self,openai_config:dict=None,system_prompt:str=ROUTER_SYS_PROMPT)->None:
        super().__init__(openai_config,system_prompt)
        self.name="router"
        self.agent=[]

    def _add_blocks(self,memory_agent):
        if memory_agent.is_active:
            return
        self.agent.append(memory_agent)

    def map_blocks(self,user_query:str,max_blocks:int=3)->list:
        """
        Map the user query to relevant memory agents.
        Args:
            user_query (str): The user query.
            max_blocks (int): The maximum number of blocks to map.
        Returns:
            list: A list of memory agents.
        """
        if not self.agent:
            return []
        
        summary_blocks = "\n".join(
            map(lambda idx_agent: f"""
            <summary>
                <index>{idx_agent[0]}</index>
                <content>{idx_agent[1].summary}</content>
            </summary>
            """, enumerate(self.agent))
        )
        
        user_prompt_formatted = f"""
            <query> {user_query} </query>
            <summary_list>
            {summary_blocks}
            </summary_list>
            Please provide the indices of the most relevant memory summaries to the query.
        """
        response = self.generate_response(user_prompt_formatted, max_tokens=512)
        
        # extract indices from response
        try:
            # use regex to find the content between <summary_index> and </summary_index>
            import re
            match = re.search(r'<summary_index>(.*?)</summary_index>', response, re.DOTALL)
            if match:
                indices_str = match.group(1).strip()
                indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
                # restrict the total number of indices to max_blocks
                selected_agents = [self.agent[idx] for idx in indices if 0 <= idx < len(self.agent)]
                return selected_agents[:max_blocks]
            else:
                return []
        except Exception as e:
            print(f"Error parsing router response: {e}")
            return []