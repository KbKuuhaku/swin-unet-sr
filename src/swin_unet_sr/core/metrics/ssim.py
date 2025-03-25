import cv2
import torch
import torch.nn.functional as F


def calc_ssim_batch(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate SSIM on a batch

    preds: SR images (`super_res_img`), (b, c, h, w) with RGB values from 0 to 1

    labels: HR images (`high_res_img`), (b, c, h, w) with RGB values from 0 to 1
    """
    preds = preds * 255  # (b, c, h, w)
    labels = labels * 255  # (b, c, h, w)

    MAX_PIXEL_VAL = 255

    # Constants from https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    K1 = 0.01
    K2 = 0.03
    GAUSSIAN_WINDOW_SIZE = 11
    GAUSSIAN_SIGMA = 1.5

    avg_ssim_batch = _calc_avg_ssim_gaussian(
        preds,
        labels,
        max_pixel_val=MAX_PIXEL_VAL,
        k1=K1,
        k2=K2,
        gaussian_window_size=GAUSSIAN_WINDOW_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )  # (b,)

    return avg_ssim_batch


def _calc_avg_ssim_gaussian(
    preds: torch.Tensor,
    labels: torch.Tensor,
    max_pixel_val: int,
    k1: float,
    k2: float,
    gaussian_window_size: int,
    gaussian_sigma: float,
) -> torch.Tensor:
    """
    Calculate the average of SSIM / Mean SSIM (MSSIM) on images

    a. Use gaussian kernel as the window to compute local SSIM
    b. Take the average of local SSIMs on h, w, c
    """
    c1 = (k1 * max_pixel_val) ** 2
    c2 = (k2 * max_pixel_val) ** 2
    in_channels = preds.shape[1]

    gaussian_kernel = _get_gaussian_kernel(
        gaussian_window_size,
        gaussian_sigma,
        out_channels=in_channels,
    ).to(preds.device)

    gaussian_conv2d = lambda img: F.conv2d(
        img,
        weight=gaussian_kernel,
        stride=1,
        padding=0,
        groups=in_channels,  # each input channels is convolved with its own set of filters
    )

    mu1 = gaussian_conv2d(preds)
    mu2 = gaussian_conv2d(labels)
    sigma1_square = gaussian_conv2d(preds * preds) - mu1 * mu1
    sigma2_square = gaussian_conv2d(labels * labels) - mu2 * mu2
    sigma1_2 = gaussian_conv2d(preds * labels) - mu1 * mu2

    local_ssims = _calc_local_ssims(mu1, mu2, sigma1_square, sigma2_square, sigma1_2, c1, c2)

    avg_ssim = local_ssims.mean(dim=[1, 2, 3])  # avg_ssim == mssim

    return avg_ssim


def _calc_local_ssims(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    sigma1_square: torch.Tensor,
    sigma2_square: torch.Tensor,
    sigma1_2: torch.Tensor,
    c1: float,
    c2: float,
) -> torch.Tensor:
    r"""
    Calculate local SSIMs

    inputs \mu, \sigma^2 are in (b, c, h, w)

    returns: local SSIMs with (b, c, h', w')
    """
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma1_2 + c2)
    denomiator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_square + sigma2_square + c2)
    local_ssims = numerator / denomiator

    return local_ssims


def _get_gaussian_kernel(
    window_size: int,
    sigma: float,
    out_channels: int,
) -> torch.Tensor:
    # Create a float32 gaussian kernel
    kernel_1d = cv2.getGaussianKernel(window_size, sigma, ktype=cv2.CV_32F)

    # Outer product on `kernel_1d`
    kernel = kernel_1d.reshape(-1, 1) @ kernel_1d.reshape(1, -1)

    return torch.from_numpy(kernel).expand(out_channels, 1, window_size, window_size)
