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

        self.cache_path = cache_path
        self.network_name = network_name
        self.model = model
        self.reps = reps
        self.load_cached_rdm = load_cached_rdm
        self.load_cached_hiddens = load_cached_hiddens

        self.roi_dict = roi_dict

        # Create the dataset
        self.dataset = MooneyDataset(image_paths)

        # Prepare the first set of RDM hook
        self.rep_curr = 0
        self.registered_hooks = []

    def _prepare_hooks(self) -> Dict[str, RDM]:
        if self.rep_curr:
            # Remove the last rep's hooks
            for h in self.registered_hooks:
                h.remove()
            self.registered_hooks = []

            # Reset the dataset
            self.dataset.reset()

        # Create list of RDMs
        out: Dict[str, RDM] = {}

        for roi_name, roi_module in self.roi_dict.items():
            rdm = RDM(
                cache_path=self.cache_path,
                network_name=f"{self.network_name}-{roi_name}-rep{self.rep_curr}",
                load_cached_hiddens=self.load_cached_hiddens,
                load_cached_rdm=self.load_cached_rdm,
            )

            # Keep track of the RDM
            out[roi_name] = rdm

            # Register forward hooks
            _hook = hook(rdm, self.dataset)
            self.registered_hooks.append(_hook)
            roi_module.register_forward_hook(_hook)

        # Increment the counter for rep
        self.rep_curr += 1
        return out

    def compute(self, device: Union[str, torch.device] = "cpu") -> Dict[str, ComputeOut]:
        all_rdm_out = {}

        for _ in range(self.reps):
            # Create a new set of reps
            rdm_dict = self._prepare_hooks()

            # Forward pass
            # TODO: Currently not batched
            for x in self.dataset:
                # TODO: Record this output and find output performance
                self.model(x)

            # Compute the rdms
            rdm_out_dict = {k: v.get(device) for k, v in rdm_dict.items()}
            all_rdm_out |= rdm_out_dict

        self.all_rdm_out = all_rdm_out
        return all_rdm_out

    def apply_analysis(self, func: Callable[[RDM], Any]) -> Dict[str, Any]:
        return {k: func(v) for k, v in self.all_rdm_out.items()}
