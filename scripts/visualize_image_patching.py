"""
Visualization of image patching from any image of the dataset

Usage:
uv run scripts/visualize_image_patching.py --image-dir "datasets/ffhq_lr_128/" --image-name "00000.png" --save-dir "pics"
"""

from pathlib import Path

import click
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

GRID_SIZE = (4, 4)
MARGIN = 0.2


def show_image_patching(
    image_file: Path,
    save_dir: Path,
    grid_size: tuple[int, int] = GRID_SIZE,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images = split_image(image)
    grid_w, grid_h = grid_size

    show_grids(images, save_dir / "orig_grid.png", grid_size=(grid_w, grid_h))
    show_grids(images, save_dir / "rearranged_grid.png", grid_size=(1, grid_w * grid_h))


def show_grids(
    images: np.ndarray,
    save_file: Path,
    grid_size: tuple[int, int],
    margin: float = MARGIN,
) -> None:
    """
    Reference: https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html
    """
    grid_w, grid_h = grid_size
    fig = plt.figure(figsize=(grid_h * 2, grid_w * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=grid_size, axes_pad=margin)

    for ax, img in zip(grid, images):  # type: ignore
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    print(f"Sucessfully saved the result to {save_file}!")
    fig.savefig(save_file, transparent=True)


def split_image(image: np.ndarray, grid_size: tuple[int, int] = GRID_SIZE) -> np.ndarray:
    grid_w, grid_h = grid_size
    return einops.rearrange(
        image,
        "(gh h) (gw w) c -> (gh gw) h w c",
        gh=grid_h,
        gw=grid_w,
    )


@click.command()
@click.option("--image-dir", help="Directory of low resolution")
@click.option("--image-name")
@click.option("--save-dir", help="Directory where results get saved to")
def main(image_dir: str, image_name: str, save_dir: str) -> None:
    click.echo(image_dir)
    click.echo(image_name)
    click.echo(save_dir)

    image_dir_ = Path(image_dir)

    show_image_patching(image_dir_ / image_name, save_dir=Path(save_dir))


if __name__ == "__main__":
    main()
