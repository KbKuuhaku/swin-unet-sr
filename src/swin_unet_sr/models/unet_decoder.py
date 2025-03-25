from dataclasses import dataclass

import torch
import torch.nn as nn

from .conv import BaseConv


class UnetDecoder(nn.Module):
    def __init__(
        self,
        skip_channels: list[int],
        decoder_channels: list[int],
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                UnetDecoderBlock(skip_channels, in_channels, out_channels)
                for skip_channels, in_channels, out_channels in zip(
                    skip_channels,
                    decoder_channels[:-1],
                    decoder_channels[1:],
                )
            ]
        )

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        block_out = x
        for block in self.blocks:
            x_skip = skip_connections.pop()
            # print(block_out.shape, x_skip.shape)
            block_out = block(block_out, x_skip)

        return block_out


class UnetDecoderBlock(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            padding=0,
            stride=2,
        )
        self.conv = nn.Sequential(
            BaseConv(
                in_channels=out_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BaseConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        up_x = self.upsampling(x)
        cat_x = torch.concat([up_x, x_skip], dim=1)
        up_out = self.conv(cat_x)

        expect_shape = (B, self.out_channels, H * 2, W * 2)
        assert up_out.shape == expect_shape, f"[UpConv] Expect: {expect_shape} but {up_out.shape}"

        return up_out
