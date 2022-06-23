import logging
from pathlib import Path
from typing import Any, Union, Dict, Tuple
from src.Cacheable import Cacheable
from src.HiddenState import HiddenState

logging.getLogger(__name__)

class RDM(Cacheable):
    def __init__(
        self,
        cache_path: Union[str, Path], 
        network_name: str,
        num_samples: int, 
        use_cache: bool=True
    ):
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        self.cache_path = cache_path
        self.network_name = network_name

        # Hiddens store the hiddens objects
        self._hiddens: Dict[int, HiddenState] = {}

        # Initialize parent class
        super().__init__(
            cache_path=cache_path,
            item_name=f"rdm_{network_name}",
            item=None
        )
        logging.debug(f"Initialized {str(self)}.")

    def get(
        self,
        
    ) -> Tuple[torch.Tensor, List[int]]:
        return self._loaded

    def _caclulate(self):
        pass

    def register_hiddens(
        self,
        sample_id: int,
        hidden: torch.Tensor
    ):
        pass

    def visualize(self):
        pass

    def __repr__(self):
        return f"RDM(cache_path={self.cache_path}, network_name={self.network_name})"
