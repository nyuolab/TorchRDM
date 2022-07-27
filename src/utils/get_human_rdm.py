from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import scipy.io as sio
import torch


@dataclass
class HumanRDM:
    rdms: List[torch.Tensor]
    labels: List[Union[int, str]]


def get_human_meg_rdm(source_path: Path = Path("./")) -> HumanRDM:
    meg_rdm_path = source_path / "RDMs" / "MEG_RDM.mat"
    loaded = sio.loadmat(meg_rdm_path)
    meg_rdm = loaded["MEG_RDM"]

    # TODO: Now we are averaging across subjects. Any better way?
    meg_rdm = meg_rdm.mean(-1)
    meg_rdms = [torch.tensor(meg_rdm[:, :, idx]) for idx in range(meg_rdm.shape[2])]

    timestep_label = list(range(meg_rdm.shape[-1]))

    return HumanRDM(rdms=meg_rdms, labels=timestep_label)


def get_human_fmri_rdm(source_path: Path = Path("./")) -> HumanRDM:
    fmri_rdm_path = source_path / "RDMs" / "FMRI_RDM.mat"
    loaded = sio.loadmat(fmri_rdm_path)

    fmri_rdm = loaded["FMRI_RDM"]
    fmri_rdms = [torch.tensor(fmri_rdm[:, :, idx] for idx in range(fmri_rdm.shape[2]))]
    labels = [str(name).split("'")[1] for name in loaded["ROI_labels"]]

    return HumanRDM(rdms=fmri_rdms, labels=labels)
