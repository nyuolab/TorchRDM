import random

import pytest
import torch

from src.RDM import RDM


@pytest.mark.parametrize("num_hiddens", [10, 20])
@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(2)])
class TestRDM:
    def test_rdm_cache(self, num_hiddens, network_name, tmp_path):
        assert not RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)
        RDM(cache_path=tmp_path, network_name=network_name, num_samples=num_hiddens)
        assert RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)

    @pytest.mark.parametrize("hidden_size", [32, 64, 77])
    def test_rdm_register(self, num_hiddens, network_name, hidden_size, tmp_path, close):
        rdm = RDM(cache_path=tmp_path, network_name=network_name, num_samples=num_hiddens)

        # Randomly generate tensors as hiddens to register
        generated_hiddens = [torch.randn(hidden_size) for _ in range(num_hiddens)]

        # Register all
        for hidden_idx, hidden in enumerate(generated_hiddens):
            rdm.register_hiddens(sample_id=hidden_idx, hidden=hidden)
            assert len(rdm) == hidden_idx + 1

        # Check the registered hiddens are close
        for hidden_idx, hidden in enumerate(generated_hiddens):
            assert close(hidden, rdm[hidden_idx].hidden)

    @pytest.mark.parametrize("hidden_shape", [(32, 2), (64, 9), (77, 3)])
    @pytest.mark.parametrize("num_hiddens_used", [5, 8, -1])
    def test_rdm_calculate_same(
        self, num_hiddens, network_name, hidden_shape, num_hiddens_used, tmp_path, close
    ):
        rdm1 = RDM(cache_path=tmp_path, network_name=network_name, num_samples=num_hiddens)
        rdm2 = RDM(cache_path=tmp_path, network_name=network_name, num_samples=num_hiddens)

        # Randomly generate tensors as hiddens
        hiddens_to_use = list(range(num_hiddens))
        if num_hiddens_used > 0:
            hiddens_to_use = hiddens_to_use[:num_hiddens_used]
        generated_hiddens = [torch.randn(hidden_shape) for _ in range(len(hiddens_to_use))]

        # Register for both rdm
        for hidden_idx, hidden in zip(hiddens_to_use, generated_hiddens):
            rdm1.register_hiddens(sample_id=hidden_idx, hidden=hidden)
            rdm2.register_hiddens(sample_id=hidden_idx, hidden=hidden)

        # Outputs
        out1, keys1 = rdm1.get()
        out2, keys2 = rdm2.get()

        assert keys1 == keys2
        for x1, x2 in zip(out1, out2):
            assert close(x1, x2)

    @pytest.mark.parametrize("hidden_shape", [(32, 2), (64, 9), (77, 3)])
    @pytest.mark.parametrize("num_hiddens_used", [5, 8, -1])
    def test_rdm_calculate_basic(
        self, num_hiddens, network_name, hidden_shape, num_hiddens_used, tmp_path, close
    ):
        rdm = RDM(cache_path=tmp_path, network_name=network_name, num_samples=num_hiddens)

        # Randomly generate tensors as hiddens
        hiddens_to_use = list(range(num_hiddens))
        random.shuffle(hiddens_to_use)

        if num_hiddens_used > 0:
            hiddens_to_use = hiddens_to_use[:num_hiddens_used]
        generated_hiddens = [torch.randn(hidden_shape) for _ in range(len(hiddens_to_use))]

        # Register all
        for hidden_idx, hidden in zip(hiddens_to_use, generated_hiddens):
            rdm.register_hiddens(sample_id=hidden_idx, hidden=hidden)

        # Compute
        mtx, keys = rdm.get()

        # Check the keys correct
        assert set(keys) == set(hiddens_to_use)

        # Check that the computed output is symmetric
        assert close(mtx, mtx.T)

        # Check that the values are in [0,2]
        assert not torch.logical_or(mtx > 2, mtx < 0).any()

        # Check that diagonal is 0
        assert not (torch.diagonal(mtx) != 0).any()


# TODO: A test for matrix correctness?
