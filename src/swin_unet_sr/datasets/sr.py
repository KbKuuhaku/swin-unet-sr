import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

from ..core.decorators import time_this
from ..core.progress import create_progress

IMAGE_FILE_PATTERN = "*.[jpJP][npNP][egEG]"


class ImagePairError(Exception): ...


class SRDataset(Dataset):
    def __init__(
        self,
        low_res_dir: str,
        high_res_dir: str,
        start_pos: int = 0,
        end_pos: int | None = None,
    ) -> None:
        super().__init__()

        self.low_res_dir = Path(low_res_dir)
        self.high_res_dir = Path(high_res_dir)

        self.start_pos = start_pos
        self.end_pos = end_pos if end_pos is not None else sys.maxsize

        self.low_res_imgs, self.high_res_imgs = self._load_image_pairs()

        n_low_res = len(self.low_res_imgs)
        n_high_res = len(self.high_res_imgs)

        if n_low_res != n_high_res:
            raise ImagePairError(
                f"low-resolution images ({n_low_res}) has to be equal to "
                f"high-resolution images ({n_high_res})"
            )

    def __len__(self) -> int:
        return len(self.low_res_imgs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.low_res_imgs[index], self.high_res_imgs[index]

    def _load_image_pairs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        with create_progress() as self.pbar:
            low_res_imgs = self._load_images(
                self.low_res_dir,
                desc_prefix="Loading low-resolution images",
            )
            high_res_imgs = self._load_images(
                self.high_res_dir,
                desc_prefix="Loading high-resolution images",
            )

        return low_res_imgs, high_res_imgs

    def _load_images(self, img_dir: Path, desc_prefix: str) -> list[np.ndarray]:
        start_pos = self.start_pos
        end_pos = min(self.end_pos, len(list(img_dir.glob(IMAGE_FILE_PATTERN))))
        print(f"Data Index: [{start_pos}, {end_pos}]")

        total_imgs = end_pos - self.start_pos

        task_id = self.pbar.add_task(desc_prefix, total=total_imgs)

        imgs = []
        for img_file in sorted(img_dir.glob(IMAGE_FILE_PATTERN))[start_pos:end_pos]:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgs.append(img)

            self.pbar.update(
                task_id,
                description=f"[yellow]{desc_prefix} {img_file}",
                advance=1,
            )

        return imgs


# @time_this
def collate_fn(
    processor: ViTImageProcessor,
    batch: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    low_res_batch = []
    high_res_batch = []

    for low_res_img, high_res_img in batch:
        low_res_batch.append(low_res_img)
        high_res_batch.append(high_res_img)

    low_res_batch = processor(
        low_res_batch,
        return_tensors="pt",
        do_resize=False,
    )["pixel_values"]
    high_res_batch = processor(
        high_res_batch,
        return_tensors="pt",
        do_normalize=False,
        do_resize=False,
    )["pixel_values"]

    return low_res_batch.cuda(), high_res_batch.cuda()
