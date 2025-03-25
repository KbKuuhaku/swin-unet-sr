from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class Evaluator(Protocol):
    @torch.no_grad
    def evaluate(
        self,
        epoch: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> float: ...


class CheckpointSaver(Protocol):
    def save_checkpoint(self, model: nn.Module) -> Path: ...


class SRTrainer:
    def __init__(
        self,
        tb_writer: SummaryWriter,
        pbar: Progress,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.tb_writer = tb_writer
        self.pbar = pbar
        self.evaluator = evaluator

        self.log = self.pbar.console.log

    def train(
        self,
        epochs: int,
        model: nn.Module,
        optimizer: opt.Optimizer,
        scheduler: lrs.ReduceLROnPlateau,
        loss_fn: nn.Module,
        saver: CheckpointSaver,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        model.train()
        self.global_step = 0
        self.best = 0.0
        self.running_loss = 0.0

        for epoch in range(epochs):
            self._train_epoch(epoch, model, optimizer, loss_fn, scheduler, train_dataloader)

            if not self.evaluator:
                continue

            metric = self.evaluator.evaluate(epoch, model, train_dataloader, val_dataloader)

            # Learning rate tuning
            scheduler.step(metric)

            if metric >= self.best:
                self.best = metric
                ckpt_file = saver.save_checkpoint(model)
                self.log(f"Current best: {self.best:.2f} | Successfully saved to {ckpt_file}...")

        model.eval()

    def _train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: opt.Optimizer,
        loss_fn: nn.Module,
        scheduler: lrs.LRScheduler,
        train_dataloader: DataLoader,
    ) -> None:
        desc_prefix = f"[green]Training on Epoch {epoch}"
        task_id = self.pbar.add_task(desc_prefix, total=len(train_dataloader))

        for low_res_img, high_res_img in train_dataloader:
            # Forward
            super_res_img = model(low_res_img)
            loss = loss_fn(super_res_img, high_res_img)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Real-time records
            cur_lr = scheduler.get_last_lr()[0]
            self.running_loss += loss.item()
            avg_loss = self.running_loss / (self.global_step + 1)

            self.pbar.update(
                task_id,
                advance=1,
                description=f"{desc_prefix} | Loss = {avg_loss:.4f} | LR = {cur_lr:.1e}",
            )
            self.tb_writer.add_scalar("AvgLoss/step", avg_loss, global_step=self.global_step)

            self.global_step += 1
