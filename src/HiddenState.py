import torch
from src.Cacheable import Cacheable
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

        # Initialize parent class
        super().__init__(
            cache_path=cache_path,
            item_name=f"hiddens_{network_name}_{sample_id}",
            item=hiddens
        )

    @property
    def hiddens(self) -> torch.Tensor:
        logging.debug(f"Loading hidden state of network {self.network_name} for sample {sample_id}.")
        return super().item

    @item.setter
    def hiddens(self, new_hidden: torch.Tensor) -> None:
        logging.debug(f"Updating hidden state of network {self.network_name} for sample {sample_id}.")
        super().item = new_hidden

