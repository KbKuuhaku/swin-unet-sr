import importlib
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch
import torch.nn as nn


class Upsampler(Protocol):
    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class UpsamplerConfig:
    name: str

    _MODULE: ClassVar[str] = "swin_unet_sr.models.upsampler"

    def load_sampler(self, in_channels: int, out_channels: int, scale: int) -> Upsampler:
        sampler_cls = getattr(importlib.import_module(self._MODULE), self.name)

        return sampler_cls(in_channels=in_channels, out_channels=out_channels, scale=scale)


class LightweightUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * (scale * scale),
                kernel_size=3,
                padding=1,
                bias=True,
            ),  # (B, C * r^2, H, W)
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
