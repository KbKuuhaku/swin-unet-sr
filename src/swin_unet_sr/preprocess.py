import torch


def normalize(
    img: torch.Tensor,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
) -> torch.Tensor:
    return (img - img_mean) / img_std


def denormalize(
    normalized_img: torch.Tensor,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
) -> torch.Tensor:
    return normalized_img * img_std + img_mean
