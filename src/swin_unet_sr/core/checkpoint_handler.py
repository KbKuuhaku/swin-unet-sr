from pathlib import Path

import torch
import torch.nn as nn


class CheckpointHandler:
    _CKPT_NAME: str = "best.pt"

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, model: nn.Module) -> None:
        state_dict = torch.load(self.ckpt_dir / self._CKPT_NAME)
        model.load_state_dict(state_dict)

    def save_checkpoint(self, model: nn.Module) -> Path:
        ckpt_file = self.ckpt_dir / self._CKPT_NAME
        torch.save(model.state_dict(), ckpt_file)

        return ckpt_file
