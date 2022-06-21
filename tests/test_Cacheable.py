from src.Cacheable import Cacheable
import pytest
import torch
import numpy as np
import pickle

def write_file(item, cache_path, filename):
    with (cache_path/filename).open('wb') as f:
        pickle.dump(item, f)


def load_file(cache_path, filename):
    with (cache_path/filename).open('rb') as f:
        out = pickle.load(f)
    return out


class TestInitialize:
    def test_missing_cache_path(self, tmp_path):
        # Test missing cache dir
        with pytest.raises(ValueError) as einfo:
            c = Cacheable(
                item=1,
                item_name="test"
            )
        assert "Cache path" in str(einfo.value)

    def test_missing_item_name(self, tmp_path):
        # Test missing item name
        with pytest.raises(ValueError) as einfo:
            c = Cacheable(
                item=1,
                cache_path=tmp_path
            )
        assert "Item name" in str(einfo.value)

    def test_missing_cache(self, tmp_path):
        with pytest.raises(ValueError) as einfo:
            c = Cacheable(
                cache_path=tmp_path,
                item_name="test"
            )
        assert "When item input is None" in str(einfo.value)


# TODO: Add cases of tensors on different devices, w or w/o gradient?
@pytest.mark.parametrize("item",
    ["1", 1, dict(wow=42), torch.randn(2,2), np.random.normal(2,2)]
)
@pytest.mark.parametrize("item_name",
    ["string", "int", "dict", "tensor", "numpy"]
)
class TestData:
    def test_with_input(
        self,
        tmp_path,
        item,
        item_name,
        close
    ):
        c = Cacheable(
            item=item,
            cache_path=tmp_path,
            item_name=item_name
        )

        # check the items can be accessed
        assert close(c.item, item)

        # check the items are cached 
        assert (tmp_path/item_name).exists()

        # check we can update the values
        c.item = 0
        assert close(c.item, 0)
        assert close(load_file(tmp_path, item_name), 0)

    def test_without_input(
            self,
            tmp_path,
            item,
            item_name,
            close
        ):
        # Manually cache the inputs
        write_file(item, tmp_path, item_name)


        # Instantiate without the item input
        c = Cacheable(
            cache_path=tmp_path,
            item_name=item_name
        )

        # Check we have the correct data loaded
        assert close(c.item, item)

