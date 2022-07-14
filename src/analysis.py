from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.manifold import trustworthiness

from src.utils import dim_reduce
from src.utils import get_phase_sub_rdm as gpsr


def intra_phase_mean_dissimilarity(rdm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Find the mean intra phase dissimilarity.

    Parameters
    ----------
    rdm : torch.Tensor
        The RDM matrix.

    Returns
    -------
    Dict[str, torch.Tensor]
        The pre, post, and gray phase mean dissimilarities.
    """
    return {p: gpsr(rdm, p).mean() for p in ["pre", "post", "gray"]}


def inter_phase_mean_dissimilarity(rdm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Find the mean inter phase dissimilarity.

    Parameters
    ----------
    rdm : torch.Tensor
        The RDM matrix.

    Returns
    -------
    Dict[str, torch.Tensor]
        The paired pre, post, and gray phase mean dissimilarities.
    """
    phase_pairs = ["-".join(pair) for pair in combinations(["pre", "post", "gray"], 2)]
    return {p: gpsr(rdm, p).diag().mean() for p in phase_pairs}


def preservation_index(rdm: torch.Tensor) -> float:
    """Find the preservation index. Note that this is approximate in two ways:
        1. The z-transform is inaccurate since we are using Spearman, not Pearson's correlation
        2. We don't consider labels

    Parameters
    ----------
    rdm : torch.Tensor
        The RDM matrix.

    Returns
    -------
    float
        The preservation index.
    """
    pre_post = gpsr(rdm, "pre-post")
    z_score = torch.arctanh(pre_post)

    diag = z_score.diag().mean()
    z_score.fill_diagonal_(0)
    off_diag = z_score.sum() / (33**2 - 33)

    return ((diag - off_diag) / off_diag).item()


def minimum_dimension(
    list_rdm: List[torch.Tensor],
    range_dimension: Tuple[int, int] = (2, 20),
    tolerance_trustworthy: float = 0.9,
    reducer_args: Optional[Dict[Any, Any]] = None,
) -> List[int]:

    """Find the minimum dimension required to describe each RDM.

    Parameters
    ----------
    list_rdm : List[torch.Tensor]
        The list of RDM to evaluate.
    range_dimension : Tuple[int, int]
        The lower and upper dimension to consider.
    tolerance_trustworthy : float
        The threshold of trustworthiness to be considered good fit.
    reducer_args : Optional[Dict[Any, Any]]
        The additional arguments to the reducer.

    Returns
    -------
    List[int]
        The list of dimensions required.
    """
    if reducer_args is None:
        reducer_args = {}

    # Initialize ouputs
    outputs = [-1] * len(list_rdm)

    for rdm_idx, rdm in enumerate(list_rdm):
        for dim_try in reversed(range(*range_dimension)):
            # TODO: Add basic sweeping of MDS parameters

            # Reduce dimension and then find trustworthiness
            reduced = dim_reduce(rdm=rdm, dim=dim_try, **reducer_args)
            t = trustworthiness(rdm, reduced)

            if t > tolerance_trustworthy:
                outputs[rdm_idx] = dim_try

    return outputs
