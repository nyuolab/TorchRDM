from itertools import product
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.RDM import RDM

image_size = 128
n_img_per_phase = 6


def get_img_sequence(img_ids: torch.Tensor, img_paths: Dict[int, Path]) -> torch.Tensor:
    img_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    out = [img_transforms(Image.open(img_paths[int(img_id)])) for img_id in img_ids]
    return torch.stack(out)


class MooneyDataset(Dataset):
    def __init__(self, image_paths: Dict[int, Path]) -> None:
        self.image_paths = image_paths
        self.reset()

    def reset(self):
        N = len(self.image_paths)

        gray_start = n_img_per_phase
        post_start = 2 * n_img_per_phase

        post_img_offset = N
        gray_img_offset = 2 * N

        # Generate raw_image_indices by permuting and padding
        images = torch.arange(N)[torch.randperm(N)]
        to_pad = N % (2 * n_img_per_phase)
        extra_images = images[torch.randperm(N)[:to_pad]]
        raw_image_indices = torch.cat([images, extra_images]).reshape(-1, 2 * n_img_per_phase)

        # Repeat pre phase to get the post phase
        raw_image_indices = torch.cat(
            [raw_image_indices, raw_image_indices[:, :n_img_per_phase]], 1
        )

        # Generate image_indices by offsetting raw index based on phase
        raw_image_indices[:, gray_start:post_start] = (
            raw_image_indices[:, gray_start:post_start] + gray_img_offset
        )
        raw_image_indices[:, post_start:] = raw_image_indices[:, post_start:] + post_img_offset
        self.image_indices = raw_image_indices

        # Generate file name library
        file_names = self.image_indices
        self.file_names = file_names % N

    def __len__(self) -> int:
        return len(self.image_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return get_img_sequence(self.file_names[idx], self.image_paths)


class hook:
    def __init__(self, rdm: RDM, dataset: MooneyDataset) -> None:
        self.rdm = rdm
        self.dataset = dataset

        # Keep track of how many batches are off
        self.reset()

    def reset(self):
        self.batch_pointer: int = 0

    def __call__(self, _unused1, _unused2, output: torch.Tensor) -> None:
        if isinstance(output, tuple):
            output = output[0]

        indices = self.dataset.image_indices
        T = indices.shape[1]
        B = output.shape[0] // T
        output = output.reshape(B, T, -1)

        # Iterate through all batches and register the hidden
        for idx1, idx2 in product(range(B), range(T)):
            self.rdm.register_hiddens(
                int(indices[self.batch_pointer, idx2]), hidden=output[idx1, idx2]
            )
            self.batch_pointer += 1
