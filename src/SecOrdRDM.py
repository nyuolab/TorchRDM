import logging
from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch

from src.RDM import RDM

logging.getLogger(__name__)


class SecOrdRDM(RDM):
    def __init__(
        self,
        cache_path: Union[str, Path],
        network_names: Tuple[str, str],
        num_samples: int,
        sim_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        load_cached_hiddens: bool = True,
        load_cached_rdm: bool = True,
        subset_samples: List[int] = None,
    ) -> None:
        super().__init__(
            cache_path=cache_path,
            network_name=",".join(network_names),
            num_samples=num_samples,
            sim_func=sim_func,
            load_cached_hiddens=load_cached_hiddens,
            load_cached_rdm=load_cached_rdm,
            subset_samples=subset_samples,
        )

        # Initialize a mapping from (source, roi_name) to index
        self.mapping = []

    def _get_hidden_keys(self):
        return self.mapping

    def register_rdm(self, roi_name: str, rdm: Union[torch.Tensor, RDM], source: str):
        """Register an RDM to track with this Second order RDM.

        Parameters
        ----------
        roi_name : Union[int,str]
            The id to associate with this rdm.
        rdm : Union[torch.Tensor, HiddenState]
            The rdm matrix to register, or an RDM object
        """
        new_name = f"{source}-{roi_name}"
        if new_name not in self.mapping:
            self.mapping.append(new_name)
        idx = self.mapping.index(new_name)

        # If we are not directly provided an rdm matrix
        if isinstance(rdm, RDM):
            rdm = rdm.get()[0]

        # Register a modified name
        super().register_hiddens(sample_id=idx, hidden=rdm)
