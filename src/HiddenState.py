from src.Cacheable import Cacheable
import torch
from pathlib import Path
from typing import Bool
import logging
logging.getLogger(__name__)

class HiddenState(Cacheable):
    def __init__(
        self,
        cache_path:Union[str, Path],
        network_name:str,
        sample_id:int,
        hiddens:torch.Tensor,
    ):
        self.cache_path = cache_path
        self.network_name = network_name
        self.sample_id = sample_id

        # Preprocess the tensor for storage
        hiddens = self.preprocess(hiddens)

        # Initialize parent class
        super().__init__(
            cache_path=cache_path,
            item_name=f"hiddens_{network_name}_{sample_id}",
            item=hiddens
        )
    logging.debug(f"Initialized {str(self)}.")

    @staticmethod
    def preprocess(data: torch.Tensor) -> torch.Tensor:
        data = data.detach()
        data = data.to('cpu')
        return data

    @property
    def hiddens(self) -> torch.Tensor:
        logging.debug(f"Loading hidden state of {str(self)}.")
        return super().item

    @item.setter
    def hiddens(self, new_hidden: torch.Tensor) -> None:
        logging.debug(f"Updating hidden state of {str(self)}.")
        new_hidden = self.preprocess(new_hidden)
        super().item = new_hidden

    def __repr__(self):
        return f"HiddenState(net={self.network_name}, sample={self.sample_id})"

