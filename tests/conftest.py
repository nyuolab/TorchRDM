import pytest
import torch
import numpy as np
import pickle

@pytest.fixture
def write():
    def write_file(item, cache_path, filename):
        with (cache_path/filename).open('wb') as f:
            pickle.dump(item, f)
    return write_file

@pytest.fixture
def read():
    def load_file(cache_path, filename):
        with (cache_path/filename).open('rb') as f:
            out = pickle.load(f)
        return out
    return load_file

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

