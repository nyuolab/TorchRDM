import torch


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    # From https://discuss.pytorch.org/t/spearmans-correlation/91931/6
    tmp = x.argsort(-1)
    ranks = torch.zeros_like(tmp, device=tmp.device)
    a = torch.arange(x.shape[-1], device=tmp.device)
    for idx in range(tmp.shape[0]):
        ranks[idx, tmp[idx]] = a
    return ranks


def spearmanr(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """Calculates a batched spearman correlation.

    Parameters
    ----------
    x : torch.Tensor
        The from feature of shape [N, C]
    y : torch.Tensor
        The to feature of shape [N, C]

    Returns
    -------
    torch.Tensor
        The output of shape [N,]
    """
    if y is None:
        y = x

    if len(x.shape) > 2:
        x = x.flatten(1)
        y = y.flatten(1)

    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(1)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=-1)
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)
