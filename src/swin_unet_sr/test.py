from dataclasses import dataclass
from pathlib import Path

import rich
import torch.nn as nn
from torch.utils.data import DataLoader

from .core.checkpoint_handler import CheckpointHandler
from .core.progress import create_progress
from .datasets.config import DataConfig
from .datasets.sr import SRDataset, collate_fn
from .evaluator import SREvaluator
from .log import log_subset_info
from .models.swin_unet_sr import SwinUnetSR, SwinUNetSRConfig


@dataclass
class TestConfig:
    name: str
    batch_size: int
    data: DataConfig

    @property
    def ckpt_dir(self) -> Path:
        return Path("ckpts") / self.name

    @property
    def tb_dir(self) -> Path:
        return Path("runs") / self.name


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_ffhq(config: TestConfig) -> None:
    # Load model
    model_config = SwinUNetSRConfig.from_folder(config.ckpt_dir)
    model = SwinUnetSR(config=model_config)

    ckpt_handler = CheckpointHandler(config.ckpt_dir)
    ckpt_handler.load_checkpoint(model)
    model = model.cuda()

    print(f"Total parameters of SwinUnetSR: {count_parameters(model)}")
    print(f"Total parameters of encoder: {count_parameters(model.encoder)}")
    print(f"Total parameters of decoder: {count_parameters(model.decoder)}")

    # Load dataset
    print("Loading image processor...")
    processor = model_config.load_processor()
    print(processor)

    test_dataset = SRDataset(
        config.data.low_res_dir,
        config.data.high_res_dir,
        config.data.start_pos,
        config.data.end_pos,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        drop_last=True,  # make sure each batch has the same size
        collate_fn=lambda x: collate_fn(processor, x),
    )
    log_subset_info("testing", test_dataset, test_dataloader)

    with create_progress() as pbar:
        evaluator = SREvaluator(pbar=pbar)
        record = evaluator.evaluate_phase(
            "testing set",
            epoch=0,
            model=model,
            dataloader=test_dataloader,
        )

        rich.print(record)
