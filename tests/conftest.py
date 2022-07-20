import pickle

import numpy as np
import pytest
import torch


@pytest.fixture
def write():
    def write_file(item, cache_path, filename):
        with (cache_path / filename).open("wb") as f:
            pickle.dump(item, f)

    return write_file


@pytest.fixture
def read():
    def load_file(cache_path, filename):
        with (cache_path / filename).open("rb") as f:
            out = pickle.load(f)
        return out

    return load_file


@pytest.fixture
def close():
    """Simple helper fixture for determining two things are close."""

    def are_close(x, y):
        assert type(x) == type(y)
        if isinstance(x, torch.Tensor):
            return torch.allclose(x, y)
        elif isinstance(x, np.ndarray):
            return np.allclose(x, y)
        else:
            return x == y

    return are_close


@pytest.fixture
def valid_rdm(close):
    def valid_func(compute_out, hiddens_to_use=None):
        keys = compute_out.hidden_keys
        mtx = compute_out.rdm

        assert len(keys) == len(mtx)

        if hiddens_to_use is not None:
            # Check the keys correct
            assert list(keys) == sorted(list(hiddens_to_use))

        # Check that the computed output is symmetric
        assert close(mtx, mtx.T)

        # Check that the values are in [0,2]
        assert not torch.logical_or(mtx > 2, mtx < 0).any()

        # Check that diagonal is 0
        assert not (torch.diagonal(mtx) != 0).any()

    return valid_func
