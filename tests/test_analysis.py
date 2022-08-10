from itertools import combinations

import pytest
import torch
from src.torchrdm.analysis import (
    inter_phase_mean_dissimilarity,
    intra_phase_mean_dissimilarity,
    minimum_dimension,
    preservation_index,
)
from src.torchrdm.RDM import RDM


def generate_rdm(num_hiddens, hiddens_shape, network_name, cache_path):
    # Generate num_hiddens hidden states
    generated = [torch.randn(hiddens_shape) for _ in range(num_hiddens)]

    # Create a RDM object
    rdm = RDM(cache_path=cache_path, network_name=network_name)

    # Register all
    for hidden_idx, hidden in enumerate(generated):
        rdm.register_hiddens(sample_id=hidden_idx, hidden=hidden)

    return rdm


# TODO: Test the utils, like get sub rdm


@pytest.mark.parametrize("num_samples", [18, 27])
@pytest.mark.parametrize("hidden_shape", [(10, 20), (20, 3)])
class TestAnalysis:
    # TODO: Add correctness check too
    def test_intra(self, tmp_path, num_samples, hidden_shape):
        rdm = generate_rdm(num_samples, hidden_shape, "network", tmp_path)
        out = intra_phase_mean_dissimilarity(rdm.get().rdm)

        for k, v in out.items():
            assert k in ["pre", "post", "gray"]
            assert v > 0 and v < 2

    def test_inter(self, tmp_path, num_samples, hidden_shape):
        rdm = generate_rdm(num_samples, hidden_shape, "network", tmp_path)
        out = inter_phase_mean_dissimilarity(rdm.get().rdm)

        keys = ["-".join(pair) for pair in combinations(["pre", "post", "gray"], 2)]
        for k, v in out.items():
            assert k in keys
            assert v > 0 and v < 2

    def test_preservation_index(self, tmp_path, num_samples, hidden_shape):
        rdm = generate_rdm(num_samples, hidden_shape, "network", tmp_path)
        preservation_index(rdm.get().rdm)

    @pytest.mark.parametrize("num_parts", [10])
    @pytest.mark.parametrize("low", [2, 4])
    @pytest.mark.parametrize("high", [4, 6])
    def test_min_dim(self, tmp_path, num_samples, hidden_shape, num_parts, low, high):
        # First collect the generated rdms
        rdms = [
            generate_rdm(num_samples, hidden_shape, f"network-{idx}", tmp_path).get().rdm
            for idx in range(num_parts)
        ]

        # Then get the min dim
        out = minimum_dimension(rdms, (low, high))

        assert len(out) == num_parts
        for x in out:
            assert x == -1 or (x >= low and x <= high)
