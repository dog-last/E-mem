from abc import ABC

import OpenAI


class BaseAgent(ABC):
    def __init__(self,openai_config:dict=None,system_prompt:str="You are a helpful assistant.")->None:
        if openai_config is None:
            raise NotImplementedError("Other types of LLM provider is not supported yet.")
        
        self.openai_config=openai_config
        self.system_prompt=system_prompt
        self.llm = OpenAI(**self.openai_config)

    def generate_response(self,question:str,max_tokens:int=1024)->str:
        response = self.llm.chat.completions.create(
            model=self.openai_config.get("model","gpt-4o-mini"),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']