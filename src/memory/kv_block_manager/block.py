import uuid

import torch


class KVBlock:
    def __init__(self,block_id:uuid.UUID,create_timestamp:str,block_size:int=8192):
        self.block_id=block_id
        self.create_timestamp=create_timestamp
        self.store_target=f"./kv_store_data/kv_cache_{self.block_id}_{self.create_timestamp}.pt"
        # block size measures the tokes that's stored instead of number of informations/chats
        # So the context window size should be at least three times of the block size
        self.block_size=block_size
        self.block_used=0   # calc the used tokens
        self.chunk_num=0    # log the number of chunks stored in this block
        # initialize the kv cache block file
        torch.save({}, self.store_target)


    def save_cache(self,cache_state:dict,total_new_token:int):
        """
        Store the new kv cache to the block file.
        If the block is empty, just save the kv cache.
        If the block is not empty, append the new kv cache to the existing kv cache.
        Args:
            kv_cache (dict): A dict to be stored that contains the kv cache, global_offset and other information(the create step will be handled elsewhere)
            total_new_token (int): The total number of new tokens while creating the new kv cache consumed, used to analyze whether the block is full.
        
        Returns:
            bool: True if the block is full after saving and a new one needs to be created, False otherwise.
        """
        assert isinstance(cache_state, dict), "cache_state must be a dict"
        assert total_new_token > 0, "total_new_token must be positive"
        for key in cache_state.keys():
            assert key in ['global_offset','saved_chunks','chunk_number','model_id'],"Unknown key found in the cache_state"
        torch.save(cache_state, self.store_target)
        self.block_used += total_new_token
        self.chunk_num=cache_state.get('chunk_number',0)
        if self.block_used >= self.block_size:
            return True
        else:
            return False

    def load_cache(self):
        """Load cache state from disk."""
        return torch.load(self.store_target, weights_only=False)
    
    def is_full(self):
        """Check if block is full."""
        return self.block_used >= self.block_size



        

