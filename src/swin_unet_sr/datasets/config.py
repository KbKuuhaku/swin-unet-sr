from dataclasses import dataclass


@dataclass
class DataConfig:
    low_res_dir: str
    high_res_dir: str
    start_pos: int = 0
    end_pos: int | None = None
