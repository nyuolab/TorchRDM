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
        sample_id: Union[int, str],
        hidden: torch.Tensor,
        load_cached_hidden: bool = True,
    ):
        """The class to store hidden states.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store cache.
        network_name : str
            The network's name.
        sample_id : int
            The sample id to associate with a hidden state.
        hidden : torch.Tensor
            The hidden state. Must not be None when we don't allow cache.
        load_cached_hidden : bool
            Whether to allow loading cached hidden states instead of using the provided one.
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        self.cache_path = cache_path
        self.network_name = network_name
        self.sample_id = sample_id

        # Preprocess the tensor for storage
        hidden = self._preprocess(hidden)

        # If we want to use cache and a cached version exists, load it
        if load_cached_hidden and self.is_cached(cache_path, network_name, sample_id):
            hidden = None
        super().__init__(cache_path=cache_path, item_name=str(self), item=hidden)
        logging.debug(f"Initialized {str(self)}.")

    @staticmethod
    def _preprocess(data: torch.Tensor) -> torch.Tensor:
        """Preprocess tensor for cacheing. Removes the gradient and moves it to
        cpu.

        Parameters
        ----------
        data : torch.Tensor
            The tensor to be preprocessed.

        Returns
        -------
        torch.Tensor
            The preprocessed tensor.
        """
        if data is not None:
            data = data.detach()
            data = data.to("cpu")
        return data

    @staticmethod
    def is_cached(cache_path: Union[str, Path], network_name: str, sample_id: int) -> bool:
        """Detects whether a cached object exists.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store cache.
        network_name : str
            The network's name.
        sample_id : int
            The sample id to associate with the hidden state.

        Returns
        -------
        bool
            Whether a cache exists for this specific hidden state.
        """
        return Cacheable.is_cached(cache_path, HiddenState._format_name(network_name, sample_id))

    @staticmethod
    def _format_name(network_name: str, sample_id: int) -> str:
        """Generate a name for this hiddens object.

        Parameters
        ----------
        network_name : str
            The network's name.
        sample_id : int
            The sample id to associate wwith the hidden state.

        Returns
        -------
        str
            The name for this hidden state.
        """
        return f"HiddenState(net={network_name}, sample={sample_id})"

    @property
    def hidden(self) -> torch.Tensor:
        """Getter for the 'hidden' property.

        Returns
        -------
        torch.Tensor
            The loaded 'hidden' property associated with this hiddens object.
        """
        logging.debug(f"Loading hidden state of {str(self)}.")
        return super().item

    @hidden.setter
    def hidden(self, new_hidden: torch.Tensor) -> None:
        """Setter for the 'hidden' property.

        Parameters
        ----------
        new_hidden : torch.Tensor
            The new hidden state to associate with this hiddens object.
        """
        logging.debug(f"Updating hidden state of {str(self)}.")
        new_hidden = self._preprocess(new_hidden)
        super().item = new_hidden

    def __repr__(self):
        return self._format_name(self.network_name, self.sample_id)
