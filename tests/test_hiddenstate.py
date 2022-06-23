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
    def test_tensor_w_grad(self, tmp_path, hidden, hidden_id, network_name):
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

    def test_tensor_is_cached(self, tmp_path, hidden, hidden_id, network_name):
        hidden = hidden.clone()

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

    def test_tensor_is_not_cached(self, tmp_path, hidden, hidden_id, network_name):
        hidden = hidden.clone()

        # Initialize hidden object
        HiddenState(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id,
            hidden=hidden,
            load_cached_hidden=False
        )

        assert not HiddenState.is_cached(
            cache_path=tmp_path,
            network_name=network_name,
            sample_id=hidden_id
        )
