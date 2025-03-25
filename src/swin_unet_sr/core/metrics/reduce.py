import torch


def mean_reduce(metric_outs: torch.Tensor) -> float:
    return metric_outs.mean(dim=0).item()
