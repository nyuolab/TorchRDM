import torch

phase_dict = {"pre": (0, 33), "post": (33, 66), "gray": (66, 99)}


def get_phase_sub_rdm(rdm: torch.Tensor, phase: str) -> torch.Tensor:
    """Get the sub RDM from a complete RDM.

    Parameters
    ----------
    rdm : torch.Tensor
        The RDM matrix of size 99x99.
    phase : str
        The phase. Either one of pre, post, or gray, or a dash-delimited pair of them.

    Returns
    -------
    torch.Tensor
        The sub rdm.
    """
    phase_parsed = phase.split("-")
    idx1 = phase_dict[phase_parsed[0]]
    idx2 = phase_dict[phase_parsed[1]] if len(phase_parsed) == 1 else idx1
    return rdm[idx1[0] : idx1[1], idx2[0] : idx2[1]]
