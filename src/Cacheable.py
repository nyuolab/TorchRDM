import pickle
import torch
from typing import Any, Union
from pathlib import Path

import logging
logging.getLogger(__name__)

class Cacheable:
    def __init__(
            self,
            cache_path:Union[str, Path]=None,
            item:Any=None,
            item_name:str=None,
    ):
        """Initialize a cacheable

        Parameters
        ----------
        cache_path : Union[str, Path]
            The filename to store the cacheable object. Can't be None.
        item : Any
            The object to be stored. Must be picklable or None.
        item_name : str
            The name of this item. Can't be None.
        """

        if cache_path is None:
            raise RuntimeError("Cache path can't be None!")
        if item_name is None:
            raise RuntimeError("Item name can't be None!")

        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        self.cache_path = cache_path

        self.item_name = item_name
        self._loaded = item

        logging.debug(f"Initialized {str(self)}.")

        # If the cache exists, and item is None, load it
        if item is None:
            p = self.cache_path/self.item_name
            if not p.exists():
                raise RuntimeError(f"When item input is None, a cache at {p} must exist!")
            logging.info(f"No item input, loading cached file at {p}.")
            self._load()
        else:
            logging.debug(f"Got item input, caching to {p}.")
            self._cache()

    @property
    def item(self):
        """The item property. Will load if we don't have cached."""
        logging.debug("Loading item property.")
        return self._loaded

    @item.setter
    def item(self, new_item):
        if new_item != self._loaded:
            logging.debug("Updating item property.")
            self._loaded = value
            self._cache()

    def _load(self):
        # TODO: Add checks for empty path
        with open(self.cache_path / self.item_name, 'rb') as f:
            logging.debug(f"Loading {f}.")
            loaded = pickle.load(f)
        return loaded

    def _cache(self):
        # TODO: Add checks for empty path
        with open(self.cache_path / self.item_name, 'wb') as f:
            logging.debug(f"Dumping {f}.")
            pickle.dump(self._loaded, sf)

    def __repr__(self):
        return f"Cacheable(cache_path={self.cache_path}, item_name={self.item_name}, loaded={bool(self._loaded)})"
