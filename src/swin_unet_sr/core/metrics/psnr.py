import torch


def calc_mse_batch(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate PSNR on a batch

    preds: SR images (`super_res_img`), (b, c, h, w) with RGB values from 0 to 1

    labels: HR images (`high_res_img`), (b, c, h, w) with RGB values from 0 to 1
    """
    # NOTE: some implementations converts RGB to Y (luma)
    #       but it's usually used in video encoding
    # preds = convert_rgb_to_luma(preds)  # (b, 1, h, w)
    # labels = convert_rgb_to_luma(labels)  # (b, 1, h, w)
    preds = preds * 255  # (b, c, h, w)
    labels = labels * 255  # (b, c, h, w)

    avg_mse_batch = ((preds - labels) ** 2).mean(dim=[1, 2, 3])  # (b,)

    return avg_mse_batch


def calc_psnr_batch(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate PSNR on a batch

    preds: SR images (`super_res_img`), (b, c, h, w) with RGB values from 0 to 1

    labels: HR images (`high_res_img`), (b, c, h, w) with RGB values from 0 to 1
    """
    MAX_PIXEL_VAL = 255

    avg_mse = calc_mse_batch(preds, labels)
    psnr = _calc_psnr(MAX_PIXEL_VAL, avg_mse)

    return psnr


def _calc_psnr(max_pixel_val: int, mse: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Implementation of Peak-siginal-to-Noise-Ratio (PSNR)

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    return 10 * torch.log10(max_pixel_val * max_pixel_val / (mse + eps))
