import numpy as np
import pytest
import torch

from src.torchrdm.Cacheable import Cacheable


class TestInitialize:
    def test_missing_cache(self, tmp_path):
        with pytest.raises(ValueError) as einfo:
            Cacheable(cache_path=tmp_path, item_name="test")
        assert "When item input is None" in str(einfo.value)


# TODO: Add cases of tensors on different devices, w or w/o gradient?
@pytest.mark.parametrize("item", ["1", 1, dict(wow=42), torch.randn(2, 2), np.random.normal(2, 2)])
@pytest.mark.parametrize("item_name", ["string", "int", "dict", "tensor", "numpy"])
class TestData:
    def test_with_input(
        self,
        tmp_path,
        item,
        item_name,
        close,
        read,
    ):
        c = Cacheable(item=item, cache_path=tmp_path, item_name=item_name)

        # check the items can be accessed
        assert close(c.item, item)

        # check the items are cached
        assert (tmp_path / item_name).exists()

        # check we can update the values
        c.item = 0
        assert close(c.item, 0)
        assert close(read(tmp_path, item_name), 0)

    def test_without_input(self, tmp_path, item, item_name, close, write):
        # Manually cache the inputs
        write(item, tmp_path, item_name)

        # Instantiate without the item input
        c = Cacheable(cache_path=tmp_path, item_name=item_name)

        # Check we have the correct data loaded
        assert close(c.item, item)
