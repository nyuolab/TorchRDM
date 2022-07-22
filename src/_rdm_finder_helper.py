from copy import deepcopy
from itertools import product
from math import ceil
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.RDM import RDM


def get_img_sequence(
    img_ids: torch.Tensor, img_paths: Dict[int, Path], image_size: int
) -> torch.Tensor:
    img_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    out = [img_transforms(Image.open(img_paths[int(img_id)])) for img_id in img_ids]
    return torch.stack(out)


class MooneyDataset(Dataset):
    def __init__(
        self, image_paths: Dict[int, Path], image_size: int = 128, n_img_per_phase: int = 6
    ) -> None:
        self.image_size = image_size
        self.n_img_per_phase = n_img_per_phase
        self.image_paths = image_paths
        self.reset()

    def reset(self) -> None:
        N = len(self.image_paths) // 2

        gray_start = self.n_img_per_phase
        post_start = 2 * gray_start

        post_img_offset = N
        gray_img_offset = 2 * N

        # Generate raw_image_indices by permuting and padding
        images = torch.arange(N)[torch.randperm(N)] + 1
        to_pad = ceil(N / (2 * self.n_img_per_phase)) * 2 * self.n_img_per_phase - N
        extra_images = images[torch.randperm(N)[:to_pad]]

        # To find pad
        raw_image_indices = torch.cat([images, extra_images]).reshape(-1, 2 * self.n_img_per_phase)

        # Add data with reverted phase to ensure complete RDM
        pre_phase = raw_image_indices[:, : self.n_img_per_phase]
        gray_phase = raw_image_indices[:, self.n_img_per_phase :]
        reverted_phase = torch.cat([gray_phase, pre_phase], 1)
        raw_image_indices = torch.cat([raw_image_indices, reverted_phase], 0)

        # Repeat pre phase to get the post phase
        raw_image_indices = torch.cat(
            [raw_image_indices, raw_image_indices[:, : self.n_img_per_phase]], 1
        )

        # Generate image_indices by offsetting raw index based on phase
        raw_image_indices[:, gray_start:post_start] = (
            raw_image_indices[:, gray_start:post_start] + gray_img_offset
        )
        raw_image_indices[:, post_start:] = raw_image_indices[:, post_start:] + post_img_offset
        self.image_indices = raw_image_indices

        # Generate file name library
        file_names = deepcopy(self.image_indices)
        file_names[file_names > 2 * N] = 2 * N - file_names[file_names > 2 * N]
        file_names[file_names > N] = N - file_names[file_names > N]
        self.file_names = file_names

    def __len__(self) -> int:
        return len(self.image_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return get_img_sequence(self.file_names[idx], self.image_paths, self.n_img_per_phase)


class hook:
    def __init__(self, rdm: RDM, dataset: MooneyDataset, batch_size: int = 1) -> None:
        self.rdm = rdm
        self.dataset = dataset
        self.batch_size = batch_size

        # Keep track of how many batches are off
        self.reset()

    def reset(self):
        self.batch_pointer: int = 0

    def __call__(self, _unused1, _unused2, output: torch.Tensor) -> None:
        if isinstance(output, tuple):
            output = output[0]

        indices = self.dataset.image_indices
        T = indices.shape[1]
        B = output.shape[0] // T if self.batch_size != 1 else 1
        output = output.reshape(B, T, -1)

        # Iterate through all batches and register the hidden
        for idx1, idx2 in product(range(B), range(T)):
            self.rdm.register_hiddens(
                int(indices[self.batch_pointer, idx2]), hidden=output[idx1, idx2]
            )
            if idx2 == T - 1:
                self.batch_pointer += 1
