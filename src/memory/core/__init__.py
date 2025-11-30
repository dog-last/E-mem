"""
Core is a module that contains the core logic of the memory module
    - The main logic loop:
        - Handle the outer conversation managers' tool call
            - If the tool call is a fetch one, then handle the query with the full loop:
                - Use the router to lookup all the summaries, and decide which one to use (the router is defined in the router module)
                - Use the memory_agent with the decided summary attached to handle the query and input the kv cache and the query, returns useful informations
                - Handle the return of the memory agent
                - Collect all the useful informations and then Aggregator will aggregate them and return the final result
            - If the tool call is a store one, directly use the memory_agent to store the information

        - Note: no matter what, the newest agent(Need to trace) should always be activated.
"""
from .loop_handler import MemoryHandler

__all__=[
    "MemoryHandler"
]
