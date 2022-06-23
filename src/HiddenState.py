import logging
from pathlib import Path
from typing import Union

import torch

from src.Cacheable import Cacheable

logging.getLogger(__name__)


class HiddenState(Cacheable):
    def __init__(
        self,
        cache_path: Union[str, Path],
        network_name: str,
        sample_id: int,
        hidden: torch.Tensor,
    ):
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        self.cache_path = cache_path
        self.network_name = network_name
        self.sample_id = sample_id

        # Preprocess the tensor for storage
        hidden = self._preprocess(hidden)

        # Initialize parent class
        super().__init__(cache_path=cache_path, item_name=str(self), item=hidden)
        logging.debug(f"Initialized {str(self)}.")

    @staticmethod
    def _preprocess(data: torch.Tensor) -> torch.Tensor:
        if data is not None:
            data = data.detach()
            data = data.to("cpu")
        return data

    @staticmethod
    def is_cached(cache_path: Union[str, Path], network_name: str, sample_id: int):
        return super().is_cached(cache_path, self._format_name(network_name, sample_id))

    @staticmethod
    def _format_name(network_name: str, sample_id: int):
        return f"HiddenState(net={network_name}, sample={sample_id})"

    @property
    def hidden(self) -> torch.Tensor:
        logging.debug(f"Loading hidden state of {str(self)}.")
        return super().item

    @hidden.setter
    def hidden(self, new_hidden: torch.Tensor) -> None:
        logging.debug(f"Updating hidden state of {str(self)}.")
        new_hidden = self._preprocess(new_hidden)
        super().item = new_hidden

    def __repr__(self):
        return self._format_name(self.network_name, self.sample_id)
