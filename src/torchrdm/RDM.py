from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch

from .Cacheable import Cacheable
from .HiddenState import HiddenState
from .utils import spearmanr


@dataclass
class ComputeOut:
    rdm: torch.Tensor
    hidden_keys: List[Union[int, str]]


class RDM(Cacheable):
    def __init__(
        self,
        cache_path: Union[str, Path],
        network_name: str,
        sim_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        load_cached_hiddens: bool = True,
        load_cached_rdm: bool = True,
    ) -> None:

        """The class to calculate and store RDMs and corresponding hidden
        states.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store the cache.
        network_name : str
            The name of the network.
        sim_func : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The similarity function to use.
        load_cached_hiddens : bool
            Whether to use cached hidden states.
        load_cached_rdm : bool
            Whether to use cached RDM.
        subset_samples : List[int]
            If not none, will check for cache created using subsets of samples.
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        self.cache_path = cache_path
        self.network_name = network_name
        self.load_cached_rdm = load_cached_rdm
        self.load_cached_hiddens = load_cached_hiddens
        self.sim_func = sim_func

        # Hiddens store the hiddens objects
        self._hiddens: Dict[Union[int, str], HiddenState] = {}
        self.mapping: List[Union[int, str]] = []

        # If we allow use of cache and a cache exists, use it
        cached_exists = self.is_cached(cache_path, network_name)
        to_init = None if load_cached_rdm and cached_exists else [[], None]
        super().__init__(
            cache_path=cache_path,
            item_name=self._format_name(network_name),
            item=to_init,
        )

    @staticmethod
    def is_cached(
        cache_path: Union[str, Path],
        network_name: str,
    ) -> bool:
        """Check if this RDM is cached.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store the cache.
        network_name : str
            The name of the network.

        Returns
        -------
        bool
            Whether the said RDM object has already been cached.
        """
        return Cacheable.is_cached(cache_path, RDM._format_name(network_name))

    @staticmethod
    def _format_name(network_name: str, keys: Optional[List[Union[int, str]]] = None) -> str:
        """Format a name for this set of features.

        Parameters
        ----------
        network_name : str
            The name of the network.
        hidden_keys : List[Union[int,str]]
            The sample ids contained in this rdm.

        Returns
        -------
        str
            The name for a corresponding object.
        """
        if keys:
            return f"RDM(network_name={network_name})"
        else:
            return f"RDM(network_name={network_name}, hiddens={keys})"

    def _get_hidden_keys(self):
        """Simple helper to find keys of hiddens collected so far.

        Returns
        -------
        List[Union[int,str]]
            The list of sample ids.
        """
        return sorted(self._hiddens.keys())

    def __len__(self):
        return len(self._get_hidden_keys())

    def __getitem__(self, idx):
        hidden = self._hiddens.get(idx)
        if hidden is None:
            raise ValueError(f"No such hidden state {idx}")
        return hidden

    def get(self, device: Union[str, torch.device] = "cpu") -> ComputeOut:
        """Get the rdm matrix.

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to compute RDM on.

        Returns
        -------
        Tuple[torch.Tensor, List[Union[int,str]]]
            The RDM itself, and the list of sample ids used to compute it.
        """
        # Check if we updated the hiddens
        hiddens_keys = self._get_hidden_keys()
        if self.item[0] != hiddens_keys:
            out = self._caclulate(device)
            self.item = [hiddens_keys, out]
        else:
            out = self.item[1]

        return ComputeOut(out, hiddens_keys)

    def _get_size_indices(self, device):
        N = len(self._hiddens)

        # Initialize the inputs, get all lower triangle indices
        mtx = torch.zeros(N, N, device=device)
        indices_all = torch.tril_indices(N, N, -1).T
        return mtx, indices_all

    def _retrieve_hidden_index(self, idx):
        return self._hiddens[self._get_hidden_keys()[idx]].hidden

    def _caclulate(self, device: Union[str, torch.device] = "cpu"):
        """Calculates the RDM matrix.

        Parameters
        ----------
        device : str
            The device to calculate on

        Returns
        -------
        torch.Tensor
            The output matrix
        """

        # Get the indices and empty matrix
        mtx, indices_all = self._get_size_indices(device)
        minibatch = min(50, len(indices_all))

        # Compute batched output
        for start_ind in range(0, len(indices_all), minibatch):
            ind_curr = indices_all[start_ind : start_ind + minibatch]
            inputs_curr_from = []
            inputs_curr_to = []

            # Iterate through a batch of indices
            for ind_x, ind_y in ind_curr:
                inputs_curr_from.append(
                    self._retrieve_hidden_index(ind_x.item()).flatten().to(device)
                )
                inputs_curr_to.append(
                    self._retrieve_hidden_index(ind_y.item()).flatten().to(device)
                )

            # Create stacked tensors
            inputs_curr_from = torch.stack(inputs_curr_from)
            inputs_curr_to = torch.stack(inputs_curr_to)

            # Calculate batched values
            out = spearmanr(inputs_curr_from, inputs_curr_to)
            mtx[ind_curr[:, 0], ind_curr[:, 1]] = out

        # 1-r()
        mtx = 1 - (mtx + mtx.T)

        # Set diagonal to 0
        mtx = mtx.fill_diagonal_(0)

        # Restore the device to CPU
        return mtx.cpu()

    def register_hiddens(self, sample_id: Union[int, str], hidden: torch.Tensor):
        """Register a hidden state to track with this RDM.

        Parameters
        ----------
        sample_id : Union[int,str]
            The id to associate with this hidden state.
        hidden : torch.Tensor
            The hidden state tensor.
        """
        # Check if we use cache or not. If we use cache, first try to initialize
        h = HiddenState(self.cache_path, self.network_name, sample_id, hidden)

        # TODO: Warning for different sizes, pad to largest size

        # Add to the dict in self
        # TODO: This allows overwrite now. Is this desirable?
        self._hiddens[sample_id] = h

    def visualize(self):
        pass

    def __repr__(self):
        return self._format_name(self.network_name, self._get_hidden_keys())
