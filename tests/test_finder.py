from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from src.analysis import preservation_index
from src.RDMFinder import RDMFinder


def image_dict(n_img, cache_path, image_size):
    imgs: Dict[int, Path] = {}
    for idx in range(1, n_img + 1):
        for prefix in [-1, 1]:
            arr = np.random.randint(0, 256, size=(image_size, image_size))
            im = Image.fromarray(arr, mode="L")
            p = cache_path / Path(f"{prefix*idx}.png")
            im.save(p)

            # Save the image
            imgs[prefix * idx] = p
    return imgs


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.Conv2d(1, 5, 3, padding=1)
        self.m2 = nn.Conv2d(5, 5, 3, padding=1)

    def forward(self, x):
        return self.m2(self.m1(x))


@pytest.mark.parametrize("num_img", [10, 20])
@pytest.mark.parametrize("image_size", [21, 64])
@pytest.mark.parametrize("n_img_per_phase", [3, 6])
def test_finder(tmp_path, num_img, image_size, n_img_per_phase, valid_rdm):
    img_dict = image_dict(num_img, tmp_path, image_size)

    # Create a simple model
    model = SimpleModel()

    # The list of rois
    roi_dict = {
        "layer1": model.m1,
        "layer2": model.m2,
    }
    rdm_names = list(roi_dict.keys())

    # Instantiate finder
    finder = RDMFinder(
        cache_path=tmp_path,
        model=model,
        network_name="network1",
        roi_dict=roi_dict,
        image_paths=img_dict,
        image_size=image_size,
        n_img_per_phase=n_img_per_phase,
        reps=6,
    )

    # Find sample ids
    sample_ids = [idx for idx in range(1, 3 * num_img + 1)]

    out = finder.compute()
    for name, rdm in out.items():
        assert name in rdm_names
        valid_rdm(rdm, sample_ids)

    # TODO: Check that we can apply analysis
    finder.apply_analysis(preservation_index)
