from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from .core.checkpoint_handler import CheckpointHandler
from .core.progress import create_progress
from .core.trainers.sr import SRTrainer
from .datasets.config import DataConfig
from .datasets.sr import SRDataset, collate_fn
from .evaluator import SREvaluator
from .log import log_subset_info
from .models.swin_unet_sr import SwinUnetSR, SwinUNetSRConfig


@dataclass
class HyperParams:
    batch_size: int
    lr: float
    train_val_test_split: list[float]


@dataclass
class LRSchedulerConfig:
    factor: float
    patience: int


@dataclass
class TrainConfig:
    name: str
    seed: int
    epochs: int
    hyper: HyperParams
    scheduler: LRSchedulerConfig
    data: DataConfig
    model: SwinUNetSRConfig

    @property
    def ckpt_dir(self) -> Path:
        return Path("ckpts") / self.name

    @property
    def tb_dir(self) -> Path:
        return Path("runs") / self.name

    def save_model_config(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_config(self.ckpt_dir)


def train_ffhq(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    # Load dataset
    print("Loading image processor...")
    processor = config.model.load_processor()
    print(processor)
    dataset = SRDataset(
        config.data.low_res_dir,
        config.data.high_res_dir,
        config.data.start_pos,
        config.data.end_pos,
    )
    train_subset, val_subset, test_subset = random_split(
        dataset,
        config.hyper.train_val_test_split,
    )

    train_dataloader = DataLoader(
        train_subset,
        batch_size=config.hyper.batch_size,
        shuffle=True,
        drop_last=True,  # make sure each batch has the same size
        collate_fn=lambda x: collate_fn(processor, x),
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=config.hyper.batch_size,
        collate_fn=lambda x: collate_fn(processor, x),
    )

    log_subset_info("training", train_subset, train_dataloader)
    log_subset_info("validation", val_subset, val_dataloader)

    # Load model, optimizer and scheduler
    print("Loading model...")
    model = SwinUnetSR(config.model).cuda()
    optimizer = opt.Adam(model.parameters(), lr=config.hyper.lr)
    scheduler = lrs.ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
    )
    loss_fn = nn.L1Loss()

    # Load TensorBoard writer
    tb_writer = SummaryWriter(config.tb_dir)

    # Load checkpoint handler
    ckpt_handler = CheckpointHandler(config.ckpt_dir)

    with create_progress() as pbar:
        evaluator = SREvaluator(
            tb_writer=tb_writer,
            pbar=pbar,
        )
        trainer = SRTrainer(
            tb_writer=tb_writer,
            pbar=pbar,
            evaluator=evaluator,
        )

        config.save_model_config()
        trainer.train(
            config.epochs,
            model,
            optimizer,
            scheduler,
            loss_fn=loss_fn,
            saver=ckpt_handler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
