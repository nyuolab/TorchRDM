from itertools import combinations
from typing import Dict

import torch

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
