import logging
from pathlib import Path
from typing import Any, Union, Dict, Tuple, Callable
from src.Cacheable import Cacheable
from src.HiddenState import HiddenState

logging.getLogger(__name__)

class RDM(Cacheable):
    def __init__(
        self,
        cache_path: Union[str, Path], 
        network_name: str,
        num_samples: int, 
        sim_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None,
        load_cached_hiddens: bool=True,
        load_cached_rdm: bool=True,
        subset_samples: List[int]=None
    ) -> None:

        """The class to calculate and store RDMs and corresponding hidden states.

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
        super().__init__(
            cache_path=cache_path,
            item_name=f"rdm_{network_name}",
            item=to_init
        )
        logging.debug(f"Initialized {str(self)}.")

    @staticmethod
    def is_cached(
            cache_path: Union[str, Path],
            network_name: str,
            sample_ids: List[int]=None,
            num_samples: int=None
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
        return super().is_cached(cache_path, self._format_name(network_name, hidden_keys))

    @staticmethod
    def _format_name(
        network_name: str,
        hidden_keys: List[int]
    ) -> str:
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

    def get(
        self,
        device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, List[int]]:
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
        hidden_keys = self._get_hidden_keys()
        if super().item[0] != hiddens_keys:
            logging.debug("Updating existing RDM...")
            out = self._caclulate()
            super().item = [hiddens_keys, out]
        else:
            logging.debug("Loading existing RDM...")
            out = super().item[1]

        return out, hiddens_keys

    def _caclulate(self):
        pass

    def register_hiddens(
        self,
        sample_id: int,
        hidden: torch.Tensor
    ):
        # Check if we use cache or not. If we use cache, first try to initialize 
        if self.use_cache:
            if HiddenState.is_cached(self.cache_path, self.network_name, sample_id):
                h = HiddenState(self.cache_path, self.network_name, sample_id)
            else:
                h = HiddenState(self.cache_path, self.network_name, sample_id, hidden)

        # Add to the dict in self
        # TODO: This allows overwrite now. Is this desirable?
        self._hiddens[sample_id] = h

    def visualize(self):
        pass

    def __repr__(self):
        return self._format_name(self.network_name, self._get_hidden_keys())

