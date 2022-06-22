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
def test_tensor_w_grad(tmp_path, hidden, hidden_id, network_name):
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


# TODO: What else to test hidden state? Currently not much is different from cacheable.
