import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import torch

from src.Cacheable import Cacheable
from src.HiddenState import HiddenState
from src.utils import spearmanr

logging.getLogger(__name__)


class RDM(Cacheable):
    def __init__(
        self,
        cache_path: Union[str, Path],
        network_name: str,
        num_samples: int,
        sim_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        load_cached_hiddens: bool = True,
        load_cached_rdm: bool = True,
        subset_samples: List[int] = None,
    ) -> None:

        """The class to calculate and store RDMs and corresponding hidden
        states.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store the cache.
        network_name : str
            The name of the network.
        num_samples : int
            The number of samples to calculate for.
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
        self._hiddens: Dict[int, HiddenState] = {}

        # If we allow use of cache and a cache exists, use it
        cached_list = subset_samples if subset_samples is not None else list(range(num_samples))
        cached_exists = self.is_cached(cache_path, network_name, cached_list)
        to_init = None if load_cached_rdm and cached_exists else [[], None]
        super().__init__(cache_path=cache_path, item_name=f"rdm_{network_name}", item=to_init)
        logging.debug(f"Initialized {str(self)}.")

    @staticmethod
    def is_cached(
        cache_path: Union[str, Path],
        network_name: str,
        sample_ids: List[int] = None,
        num_samples: int = None,
    ) -> bool:
        """Check if this RDM is cached.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The path to store the cache.
        network_name : str
            The name of the network.
        sample_ids : List[int]
            The sample ids to check for. If want the entire list, leave empty.
        num_samples : int
            The number of samples to check for. Must not be None when sample_ids is None.

        Returns
        -------
        bool
            Whether the said RDM object has already been cached.
        """
        if sample_ids is None:
            if num_samples is not None:
                sample_ids = list(range(num_samples))
            else:
                raise ValueError("Must provide either a list of sample ids or number of samples.")
        return Cacheable.is_cached(cache_path, RDM._format_name(network_name, sample_ids))

    @staticmethod
    def _format_name(network_name: str, hidden_keys: List[int]) -> str:
        """Format a name for this set of features.

        Parameters
        ----------
        network_name : str
            The name of the network.
        hidden_keys : List[int]
            The sample ids contained in this rdm.

        Returns
        -------
        str
            The name for a corresponding object.
        """
        return f"RDM(network_name={network_name}, sample_ids={hidden_keys})"

    def _get_hidden_keys(self):
        """Simple helper to find keys of hiddens collected so far.

        Returns
        -------
        List[int]
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

    def get(self, device: Union[str, torch.device] = "cuda:0") -> Tuple[torch.Tensor, List[int]]:
        """Get the rdm matrix.

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to compute RDM on.

        Returns
        -------
        Tuple[torch.Tensor, List[int]]
            The RDM itself, and the list of sample ids used to compute it.
        """
        # Check if we updated the hiddens
        hiddens_keys = self._get_hidden_keys()
        if super().item[0] != hiddens_keys:
            logging.debug("Updating existing RDM...")
            out = self._caclulate(device)
            super().item = [hiddens_keys, out]
        else:
            logging.debug("Loading existing RDM...")
            out = super().item[1]

        return out, hiddens_keys

    def _caclulate(self, device: Union[str, torch.device] = "cuda:0"):
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
        # Find the pairwise score
        inputs = [self._hiddens[k] for k in self._get_hidden_keys()]

        # Setup minibatch
        N = len(self._hiddens)
        minibatch = 50

        # Initialize the inputs, get all lower triangle indices
        mtx = torch.zeros(N, N, device=device)
        indices_all = torch.tril_indices(N, N, -1).T

        # Compute batched output
        for start_ind in range(0, len(indices_all) ** 2, minibatch):
            ind_curr = indices_all[start_ind : start_ind + minibatch]
            inputs_curr_from = []
            inputs_curr_to = []

            # Iterate through a batch of indices
            for ind_x, ind_y in ind_curr:
                inputs_curr_from.append(torch.Tensor(inputs[ind_x].hidden, device=device))
                inputs_curr_to.append(torch.Tensor(inputs[ind_y].hidden, device=device))

            # Create stacked tensors
            inputs_curr_from = torch.stack(inputs_curr_from)
            inputs_curr_to = torch.stack(inputs_curr_to)

            # Calculate batched values
            mtx[indices_all[:, 0], indices_all[:, 1]] = spearmanr(inputs_curr_from, inputs_curr_to)

        # 1-r()
        mtx = 1 - (mtx + mtx.T)

        # Set diagonal to 0
        mtx = mtx.fill_diagonal_(0)

        # Restore the device to CPU
        return mtx.cpu()

    def register_hiddens(self, sample_id: int, hidden: torch.Tensor):
        """Register a hidden state to track with this RDM

        Parameters
        ----------
        sample_id : int
            The id to associate with this hidden state.
        hidden : torch.Tensor
            The hidden state tensor.
        """
        # Check if we use cache or not. If we use cache, first try to initialize
        h = HiddenState(self.cache_path, self.network_name, sample_id, hidden)

        # Add to the dict in self
        # TODO: This allows overwrite now. Is this desirable?
        self._hiddens[sample_id] = h

    def visualize(self):
        pass

    def __repr__(self):
        return self._format_name(self.network_name, self._get_hidden_keys())
