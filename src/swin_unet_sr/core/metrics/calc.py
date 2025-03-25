from __future__ import annotations

from typing import Callable

import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Reduction = Callable[[torch.Tensor], float]


class MetricCalculator:
    def __init__(self, metric: Metric) -> None:
        self.metric = metric
        self.metric_outs = []

    def calc_batch(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        metric_out = self.metric(preds, labels)
        self.metric_outs.append(metric_out)

    def calc_all(self, reduce: Reduction) -> float:
        metric_outs = torch.cat(self.metric_outs, dim=0)
        return reduce(metric_outs)
