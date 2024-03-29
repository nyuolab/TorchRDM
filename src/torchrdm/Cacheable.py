import pickle
from pathlib import Path
from typing import Any, Union


class Cacheable:
    def __init__(
        self,
        cache_path: Union[str, Path],
        item_name: str,
        item: Any = None,
    ):
        """Initialize a cacheable.

        Parameters
        ----------
        cache_path : Union[str, Path]
            The filename to store the cacheable object.
        item : Any
            The object to be stored. Must be picklable or None.
        item_name : str
            The name of this item.
        """

        if cache_path is None:
            raise ValueError("Cache path can't be None!")
        if item_name is None:
            raise ValueError("Item name can't be None!")

        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        self.cache_path = cache_path
        self.item_name = item_name
        self._loaded = item

        p = self.cache_path / self.item_name
        if item is None:
            if not p.exists():
                raise ValueError(f"When item input is None, a cache at {p} must exist!")
            self._load()
        else:
            self._cache()

    @staticmethod
    def is_cached(cache_path: Union[str, Path], item_name: str):
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        return (cache_path / Path(item_name)).exists()

    @property
    def item(self) -> Any:
        """The item property.

        Will load if we don't have cached.
        """
        return self._loaded

    @item.setter
    def item(self, new_item: Any) -> None:
        # TODO: Add checks for identity? Equal sign comparison could throw errors.
        self._loaded = new_item
        self._cache()

    def _load(self) -> None:
        # TODO: Add checks for empty path
        with (self.cache_path / self.item_name).open("rb") as f:
            loaded = pickle.load(f)
        self._loaded = loaded

    def _cache(self) -> None:
        # TODO: Add checks for empty path
        with (self.cache_path / self.item_name).open("wb") as f:
            pickle.dump(self._loaded, f)

    def __repr__(self):
        return f"Cacheable(cache_path={self.cache_path}, item_name={self.item_name})"
