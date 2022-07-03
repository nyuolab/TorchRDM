import pytest
import torch

from src.HiddenState import HiddenState


@pytest.mark.parametrize(
    "hidden",
    [
        torch.randn(5, 5),
        torch.randn(6, 6),
        torch.randn(
            7,
        ),
    ],
)
@pytest.mark.parametrize("hidden_id", list(range(3)))
@pytest.mark.parametrize("network_name", [f"network{i}" for i in range(3)])
class TestData:
    def test_hidden_tensor_w_grad(self, tmp_path, hidden, hidden_id, network_name):
        hidden = hidden.clone()

        # Create gradient in hidden
        hidden.requires_grad = True
        hidden = hidden * torch.randn(
            1,
        )

        # Initialize hidden object
        HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
        )

    def test_hidden_tensor_is_cached(self, tmp_path, hidden, hidden_id, network_name, close):
        hidden = hidden.clone()

        # There shouldn't be a cache now
        assert not HiddenState.is_cached(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id
        )

        # Initialize hidden object
        HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
        )

        assert HiddenState.is_cached(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id
        )

        # Use a different input
        hidden = hidden * 4
        h_new = HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
        )

        # The output should be different
        assert not close(h_new.hidden, hidden)

    def test_hidden_tensor_is_not_cached(self, tmp_path, hidden, hidden_id, network_name, close):
        hidden = hidden.clone()

        # We shouldn't have a cache now
        assert not HiddenState.is_cached(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id
        )


        # Initialize hidden object
        HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
        )

        # There should be a cache here
        assert HiddenState.is_cached(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id
        )

        # Use a different input
        hidden = hidden * 4
        h_new = HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
            load_cached_hidden=False
        )

        # The output should be the same
        assert close(h_new.hidden, hidden)
