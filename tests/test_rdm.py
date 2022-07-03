import pytest

from src.RDM import RDM


@pytest.mark.parametrize("num_hiddens", list(range(2, 5)))
@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(3)])
class TestRDM:
    # Test that rdm is cached
    def test_rdm_cache_arg(self, num_hiddens, network_name, tmp_path, close):
        assert not RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)
        RDM(
            cache_path=tmp_path,
            network_name=network_name,
            num_samples=num_hiddens,
        )
        assert RDM.is_cached(tmp_path, network_name, num_samples=num_hiddens)


# Test that we can register hiddens and confirm ids

# Test that we can access hiddens

# Test that we have right length

# Test that the RDM of two identical hiddens are identical
