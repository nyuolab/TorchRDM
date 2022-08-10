from itertools import product

import pytest
import torch
from src.TorchRDM.RDM import RDM
from src.TorchRDM.SecOrdRDM import SecOrdRDM


def generate_rdm(num_hiddens, hiddens_shape, network_name, cache_path):
    # Generate num_hiddens hidden states
    generated = [torch.randn(hiddens_shape) for _ in range(num_hiddens)]

    # Create a RDM object
    rdm = RDM(cache_path=cache_path, network_name=network_name)

    # Register all
    for hidden_idx, hidden in enumerate(generated):
        rdm.register_hiddens(sample_id=hidden_idx, hidden=hidden)

    return rdm


@pytest.mark.parametrize("num_samples", [10, 20])
@pytest.mark.parametrize("hidden_shape", [(10, 20), (20, 3)])
def test_sec_ord(num_samples, hidden_shape, tmp_path, valid_rdm):
    network_names = ("network_1", "network_2")
    network_rois = ["layer1", "layer2.0"]

    # Create a SecOrdRDM object
    sec_ord_rdm = SecOrdRDM(
        cache_path=tmp_path,
        network_names=network_names,
    )

    # Keep track of generated hiddens
    generated_hiddens = {}

    # Register all rdms
    for network, roi in product(network_names, network_rois):
        # Generate a RDM object
        rdm = generate_rdm(num_samples, hidden_shape, network, tmp_path)

        # Register a generated RDM
        sec_ord_rdm.register_rdm(roi_name=roi, rdm=rdm, source=network)

        # Keep track of the current generated hidden
        generated_hiddens[(network, roi)] = rdm.get()

    # Calculate the sec ord rdm
    out = sec_ord_rdm.get()

    # Second ordr RDM is still RDM
    valid_rdm(out, [f"{net}-{roi}" for net, roi in product(network_names, network_rois)])

    # TODO: Check correctness
