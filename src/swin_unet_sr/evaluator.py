from dataclasses import dataclass

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .core.metrics.calc import MetricCalculator
from .core.metrics.psnr import calc_mse_batch, calc_psnr_batch
from .core.metrics.reduce import mean_reduce
from .core.metrics.ssim import calc_ssim_batch


@dataclass
class Record:
    psnr: float
    ssim: float
    mse: float


class SREvaluator:
    def __init__(
        self,
        pbar: Progress,
        tb_writer: SummaryWriter | None = None,
    ) -> None:
        self.tb_writer = tb_writer
        self.pbar = pbar
        self.log = self.pbar.console.log

    @torch.no_grad
    def evaluate(
        self,
        epoch: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> float:
        border = "=" * 40
        self.log(f"{border}Evaluation{border}")

        train_record = self.evaluate_phase("training set", epoch, model, train_dataloader)
        val_record = self.evaluate_phase("validation set", epoch, model, val_dataloader)

        self.log(f"{border}Evaluation{border}")

        self._plot_to_tensorboard(epoch, train_record, val_record)

        return val_record.psnr

    @torch.no_grad
    def evaluate_phase(
        self,
        phase: str,
        epoch: int,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Record:
        model.eval()

        task_id = self.pbar.add_task(f"[yellow]Evaluating on {phase}", total=len(dataloader))
        mse_calc = MetricCalculator(metric=calc_mse_batch)
        psnr_calc = MetricCalculator(metric=calc_psnr_batch)
        ssim_calc = MetricCalculator(metric=calc_ssim_batch)

        for low_res_img, high_res_img in dataloader:
            super_res_img = model(low_res_img)

            mse_calc.calc_batch(super_res_img, high_res_img)
            psnr_calc.calc_batch(super_res_img, high_res_img)
            ssim_calc.calc_batch(super_res_img, high_res_img)

            self.pbar.advance(task_id)

        avg_mse = mse_calc.calc_all(reduce=mean_reduce)
        avg_psnr = psnr_calc.calc_all(reduce=mean_reduce)
        avg_ssim = ssim_calc.calc_all(reduce=mean_reduce)

        self.log(
            f"Epoch {epoch} {phase:<15}"
            f"| MSE = {avg_mse:<10.2f}"
            f"| PSNR (dB) = {avg_psnr:<10.2f}"
            f"| SSIM = {avg_ssim:<10.2f}"
        )
        model.train()

        return Record(psnr=avg_psnr, ssim=avg_ssim, mse=avg_mse)

    def _plot_to_tensorboard(self, epoch: int, train_record: Record, val_record: Record) -> None:
        if self.tb_writer is None:
            return

        self.tb_writer.add_scalars(
            "PSNR/epoch",
            {
                "train": train_record.psnr,
                "val": val_record.psnr,
            },
            global_step=epoch,
        )
        self.tb_writer.add_scalars(
            "SSIM/epoch",
            {
                "train": train_record.ssim,
                "val": val_record.ssim,
            },
            global_step=epoch,
        )
        self.tb_writer.add_scalars(
            "MSE/epoch",
            {
                "train": train_record.mse,
                "val": val_record.mse,
            },
            global_step=epoch,
        )
