import pytest
import torch
import numpy as np

@pytest.fixture
def close():
    """Simple helper fixture for determining two things are close"""
    def are_close(x, y):
        assert type(x) == type(y)
        if isinstance(x, torch.Tensor):
            return torch.allclose(x, y)
        elif isinstance(x, np.ndarray):
            return np.allclose(x, y)
        else:
            return x==y
    return are_close

