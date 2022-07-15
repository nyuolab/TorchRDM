from pathlib import Path
from typing import Any, Callable, Dict, Union

import torch
import torch.nn as nn

from src._rdm_finder_helper import MooneyDataset, hook
from src.RDM import RDM, ComputeOut

image_size = 128
n_img_per_phase = 6


class RDMFinder:
    def __init__(
        self,
        cache_path: Union[str, Path],
        model: nn.Module,
        network_name: str,
        roi_dict: Dict[str, nn.Module],
        image_paths: Dict[int, Path],
        reps: int = 6,
        load_cached_hiddens: bool = True,
        load_cached_rdm: bool = True,
    ) -> None:

        self.network_name = network_name
        self.model = model
        self.reps = reps

        # Create the dataset
        self.dataset = MooneyDataset(image_paths)

        # Create list of RDMs
        self._rdm_dict: Dict[str, RDM] = {}

        for roi_name, roi_module in roi_dict.items():
            rdm = RDM(
                cache_path=cache_path,
                network_name=f"{network_name}-{roi_name}",
                load_cached_hiddens=load_cached_hiddens,
                load_cached_rdm=load_cached_rdm,
            )

            # Keep track of the RDM
            self._rdm_dict[roi_name] = rdm

            # Register forward hooks
            roi_module.register_forward_hook(hook(rdm, self.dataset))

    def compute(self, device: Union[str, torch.device] = "cpu") -> Dict[str, ComputeOut]:
        # Iterate through the reps
        # First iterate through the dataset
        # TODO: Currently not batched
        # TODO: How do we deal with multiple reps?

        return {k: v.get(device) for k, v in self._rdm_dict.items()}

    def apply_analysis(self, func: Callable[[RDM], Any]) -> Dict[str, Any]:
        return {k: func(v) for k, v in self._rdm_dict.items()}
