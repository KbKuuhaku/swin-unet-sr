import numpy as np
import torch

# Reference: https://en.wikipedia.org/wiki/YCbCr
#            Section: ITU-R BT.601 conversion
_RGB_TO_YCBCR_TRANSFORM = np.array(
    [
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214],
    ]
)

_RGB_TO_YCBCR_BIAS = np.array(
    [
        [16],
        [128],
        [128],
    ]
)


def convert_rgb_to_luma(rgb_imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB images (range: 0 - 1) to Y/luma (range: 0 - 255) images

    rgb_imgs: (b, c, h, w) RGB images

    return: (b, 1, h, w) luma images
    """
    rgb_to_luma_transform = torch.tensor(_RGB_TO_YCBCR_TRANSFORM[0], device=rgb_imgs.device)
    rgb_to_luma_transform = rgb_to_luma_transform.reshape(1, -1, 1, 1)
    rgb_to_luma_bias = torch.tensor(_RGB_TO_YCBCR_BIAS[0], device=rgb_imgs.device)

    luma = (rgb_to_luma_transform * rgb_imgs + rgb_to_luma_bias).sum(dim=1, keepdim=True)
    return luma


def convert_rgb_to_ycbcr(rgb_imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB images (range: 0 - 1) to YCbCr (range: 0 - 255) images

    rgb_imgs: (b, c, h, w) RGB images

    return: (b, c, h, w) luma images
    """
    rgb_to_ycbcr_transform = torch.tensor(_RGB_TO_YCBCR_TRANSFORM[0], device=rgb_imgs.device)
    rgb_to_ycbcr_transform = rgb_to_ycbcr_transform.reshape(1, -1, 1, 1)
    rgb_to_ycbcr_bias = torch.tensor(_RGB_TO_YCBCR_BIAS, device=rgb_imgs.device)

    ycbcr = rgb_to_ycbcr_transform * rgb_imgs + rgb_to_ycbcr_bias

    return ycbcr
