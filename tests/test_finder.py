import torch.nn as nn

from typing import Dict
import numpy as np
from PIL import Image
from pathlib import Path

from src.RDMFinder import RDMFinder

from src.analysis import (
    preservation_index,
)

import pytest


# Only do once for this module
def image_dict(n_img, path, image_size):
    imgs: Dict[int, Path]  = {}
    for idx in range(1, n_img+1):
        for prefix in [-1, 1]:
            arr = np.random.randint(0, 256, size=(image_size,image_size))
            im = Image.fromarray(arr, mode="L")
            p = Path(f"{prefix*idx}.png")
            im.save(path / p)

            # Save the image
            imgs[prefix*idx] = p
    return imgs

@pytest.mark.parametrize("num_img", [10, 20])
@pytest.mark.parametrize("image_size", [21, 64])
@pytest.mark.parametrize("n_img_per_phase", [3, 5])
def test_finder(tmp_path, num_img, image_size, n_img_per_phase):
    img_dict = image_dict(num_img, tmp_path, image_size)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1, 5),
        nn.Linear(5, 5),
    )

    finder = RDMFinder(
        cache_path=tmp_path,
        model=model,
        network_name="network1",
        roi_dict={
            "layer1": model[0],
            "layer2": model[1],
        },
        image_paths=img_dict,
        image_size=image_size,
        n_img_per_phase=n_img_per_phase,
        reps=6,
    )

    finder.compute()
    # Check the keys
    finder.apply_analysis(preservation_index)
