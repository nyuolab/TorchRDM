import random

import pytest
import torch

from src.torchrdm.RDM import RDM


@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(2)])
def test_rdm_cache(network_name, tmp_path):
    assert not RDM.is_cached(tmp_path, network_name)
    RDM(cache_path=tmp_path, network_name=network_name)
    assert RDM.is_cached(tmp_path, network_name)


@pytest.mark.parametrize("num_hiddens", [10, 20])
@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(2)])
class TestRDM:
    @pytest.mark.parametrize("hidden_size", [32, 64, 77])
    def test_rdm_register(self, num_hiddens, network_name, hidden_size, tmp_path, close):
        rdm = RDM(cache_path=tmp_path, network_name=network_name)

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
        rdm1 = RDM(cache_path=tmp_path, network_name=network_name)
        rdm2 = RDM(cache_path=tmp_path, network_name=network_name)

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
        out1, out2 = rdm1.get(), rdm2.get()
        out1, keys1 = out1.rdm, out1.hidden_keys
        out2, keys2 = out2.rdm, out2.hidden_keys

        assert keys1 == keys2
        for x1, x2 in zip(out1, out2):
            assert close(x1, x2)

    @pytest.mark.parametrize("hidden_shape", [(32, 2), (64, 9), (77, 3)])
    @pytest.mark.parametrize("num_hiddens_used", [5, 8, -1])
    def test_rdm_calculate_basic(
        self, num_hiddens, network_name, hidden_shape, num_hiddens_used, tmp_path, close, valid_rdm
    ):
        rdm = RDM(cache_path=tmp_path, network_name=network_name)

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
        out = rdm.get()
        valid_rdm(out, hiddens_to_use)


# TODO: A test for matrix correctness?
