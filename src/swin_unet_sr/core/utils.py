import einops
import numpy as np
import torch
from numpy.typing import NDArray


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_grid(arrs: list[NDArray], grid_size: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = grid_size

    assert (
        len(arrs) == grid_h * grid_w
    ), f"The number of images ({len(arrs)}) are not equal to the number of grids {grid_size}"

    arrs = np.stack(arrs, axis=0)  # type: ignore (b, h, w, c)

    # make grids (gh x gw)
    grid = einops.rearrange(arrs, "(gh gw) h w c -> (gh h) (gw w) c", gh=grid_h, gw=grid_w)

    return grid
