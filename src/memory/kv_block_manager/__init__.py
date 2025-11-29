"""
KV Block Manager is a module that manages all the kv cache blocks
It controls:
    - Allocating new kv cache blocks(create file and return the block id)
    - Updating the kv cache block (input the block id and the new kv cache contents added, and then it will update the kv cache file)
    - Tells if a kv cache block is full (the size is defined by params, decided by user, and may be a good recommend value need to be added to README.md)
"""