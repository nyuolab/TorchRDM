from typing import Any

import torch
from sklearn.manifold import MDS, TSNE


def dim_reduce(
    rdm: torch.Tensor, dim: int = 2, method: str = "MDS", **kwargs: Any
) -> torch.Tensor:
    """Reduce the dimensionality of the rdm.

    Parameters
    ----------
    rdm : torch.Tensor
        The RDM to reduce.
    dim : int
        The output dimensionality.
    method : str
        The method. Either TSNE or MDS for now.
    reducer_args : Any
        Any arguments to the reducer.

    Returns
    -------
    torch.Tensor
        The reduced rdm.
    """
    # Setup the reducer
    reducer = MDS if method == "MDS" else TSNE
    reducer = reducer(n_components=dim, metric="precomputed", **kwargs)

    # calculate
    rdm = reducer.fit_transform(rdm.numpy())
    return torch.tensor(rdm)
