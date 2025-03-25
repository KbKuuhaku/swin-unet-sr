import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Self

import torch
import torch.nn as nn
from transformers import AutoBackbone, AutoImageProcessor, ViTImageProcessor
from transformers.modeling_outputs import BackboneOutput

from ..core.config import parse_config
from ..preprocess import denormalize
from .unet_decoder import UnetDecoder
from .upsampler import UpsamplerConfig


@dataclass
class SwinUNetSRConfig:
    swin_checkpoint: str
    img_mean: list[float]
    img_std: list[float]
    in_channels: int
    embed_dim: int
    patch_size: int
    skip_channels: list[int]
    decoder_channels: list[int]
    upscale: int
    upsampler: UpsamplerConfig

    _CONFIG_NAME: ClassVar[str] = "config.json"

    @classmethod
    def from_folder(cls, config_dir: Path) -> Self:
        return parse_config(str(config_dir / cls._CONFIG_NAME), cls)

    @property
    def reshaped_img_mean(self) -> torch.Tensor:
        return torch.tensor(self.img_mean).reshape(1, -1, 1, 1).cuda()

    @property
    def reshaped_img_std(self) -> torch.Tensor:
        return torch.tensor(self.img_std).reshape(1, -1, 1, 1).cuda()

    def save_config(self, ckpt_dir: Path) -> None:
        # Save model config
        config_file = ckpt_dir / self._CONFIG_NAME
        print(f"Saving {config_file}...")
        with open(config_file, "w") as f:
            json.dump(
                asdict(self),
                fp=f,
                indent=4,
            )

    def load_processor(self) -> ViTImageProcessor:
        return AutoImageProcessor.from_pretrained(self.swin_checkpoint, use_fast=True)


class SwinUnetSR(nn.Module):
    """
    SwinUNetSR

    - Encoder: backbone from swin/swinv2

    - Decoder: U-Net decoder blocks, which extract deep features

    - Upsampler: upscaling deep feature extractions
    """

    def __init__(self, config: SwinUNetSRConfig) -> None:
        super().__init__()
        self.encoder = AutoBackbone.from_pretrained(config.swin_checkpoint)
        self.decoder = UnetDecoder(config.skip_channels, config.decoder_channels)

        out_channels = config.decoder_channels[-1]
        self.upsampler = config.upsampler.load_sampler(
            in_channels=out_channels,
            out_channels=config.in_channels,
            scale=config.patch_size * config.upscale,
        )

        self.img_mean = config.reshaped_img_mean
        self.img_std = config.reshaped_img_std

    @classmethod
    def from_folder(cls, config_dir: Path) -> Self:
        return cls(SwinUNetSRConfig.from_folder(config_dir))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        up_out = self.upsampler(features)
        out = denormalize(up_out, self.img_mean, self.img_std)  # denormalize it back to (0 - 1)

        return out

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_out: BackboneOutput = self.encoder(x, output_hidden_states=True)

        # NOTE: https://github.com/huggingface/transformers/blob/69bc848480d5f19a537a70ce14f09816b00cd80f/src/transformers/models/swinv2/modeling_swinv2.py#L919
        # 1. `SwinV2Backbone` stores hidden states before downsampling for decoder
        # 2. `hidden_states` also stores the **input** of encoder,
        #    which is necessary in the final residual connection
        # 3. The last layer of `SwinV2Stage` does not have downsampling
        # 4. Backbone doesn't store the hidden states after reshaping (which is weird),
        #    and I have to manually reshape them
        #
        encoder_hiddens = _reshape_hidden_states(encoder_out.hidden_states)  # type: ignore

        shallow_features = encoder_hiddens[0]
        encoder_hiddens = encoder_hiddens[1:]

        # Decoder
        skip_connections = encoder_hiddens[:-1]
        enc_features = encoder_hiddens[-1]
        deep_features = self.decoder(enc_features, skip_connections)

        # Residual connection
        features = shallow_features + deep_features

        return features


def _reshape_hidden_states(hiddens: tuple[torch.Tensor]) -> list[torch.Tensor]:
    reshaped_hiddens = []

    for hidden in hiddens:
        batch_size, area, channels = hidden.shape
        side = int(math.sqrt(area))  # suppose feature maps are square
        reshaped = hidden.reshape(batch_size, side, side, channels)
        reshaped_hiddens.append(reshaped.permute(0, 3, 1, 2))

    return reshaped_hiddens
