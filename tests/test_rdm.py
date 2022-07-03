import pytest
import torch

from src.RDM import RDM


@pytest.mark.parametrize("num_hiddens", list(range(3, 5)))
@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(3)])
class TestRDM:
    # Test that rdm is cached
    def test_rdm_cache_arg(self, num_hiddens, network_name, tmp_path):
        assert not RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)
        RDM(
            cache_path=tmp_path,
            network_name=network_name,
            num_samples=num_hiddens,
        )
        assert RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)

    # Test that we can register hiddens and confirm ids
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


# Test that the RDM of two identical hiddens are identical
