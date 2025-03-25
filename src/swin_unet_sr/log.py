from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset


def log_subset_info(phase: str, subset: Subset | Dataset, dataloader: DataLoader) -> None:
    print(f"# of {phase:<20}: {len(subset)} items, {len(dataloader)} batches")  # type: ignore
