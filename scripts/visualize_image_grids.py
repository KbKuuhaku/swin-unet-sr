"""
Visualization of low/high-resolution faces from dataset

Usage:
uv run scripts/visualize_image_grids.py --low-res-dir "datasets/ffhq_lr_256/" --high-res-dir "datasets/ffhq_hr_1024/" --save-dir "pics"
"""

import random
from pathlib import Path

import click
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np


def show_img_grid(
    low_res_folder: Path,
    high_res_folder: Path,
    save_dir: Path,
    grid_size: tuple[int, int] = (3, 10),
) -> None:
    glob_pattern = "*.[jpJP][npNP][egEG]"

    save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = grid_size[0] * grid_size[1]
    sampled_low_res_img_files = random.choices(list(low_res_folder.glob(glob_pattern)), k=n_samples)
    sampled_high_res_img_files = [high_res_folder / img.name for img in sampled_low_res_img_files]

    low_res_img_grids = make_img_grid(sampled_low_res_img_files, grid_size)
    high_res_img_grids = make_img_grid(sampled_high_res_img_files, grid_size)

    plt.xticks([])
    plt.yticks([])
    plt.imsave(save_dir / "low_res_faces.png", low_res_img_grids)
    plt.imsave(save_dir / "high_res_faces.png", high_res_img_grids)

    print(f"Sucessfully saved the result to {save_dir}!")


def make_img_grid(img_files: list[Path], grid_size: tuple[int, int]) -> np.ndarray:
    imgs = []
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return make_grid(imgs, grid_size)


def make_grid(arrs: list[np.NDArray], grid_size: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = grid_size

    assert (
        len(arrs) == grid_h * grid_w
    ), f"The number of images ({len(arrs)}) are not equal to the number of grids {grid_size}"

    arrs = np.stack(arrs, axis=0)  # type: ignore (b, h, w, c)

    # make grids (gh x gw)
    grid = einops.rearrange(arrs, "(gh gw) h w c -> (gh h) (gw w) c", gh=grid_h, gw=grid_w)

    return grid


@click.command()
@click.option("--low-res-dir", help="Directory of low resolution")
@click.option("--high-res-dir", help="Directory of high resolution")
@click.option("--save-dir", help="Directory where results get saved to")
def main(low_res_dir: str, high_res_dir: str, save_dir: str) -> None:
    click.echo(low_res_dir)
    click.echo(high_res_dir)

    low_res_folder = Path(low_res_dir)
    high_res_folder = Path(high_res_dir)

    show_img_grid(low_res_folder, high_res_folder, Path(save_dir))


if __name__ == "__main__":
    main()
