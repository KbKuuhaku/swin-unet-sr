from typing import TypeVar

import dacite
import rich

from .factories.io import read_to_dict

T = TypeVar("T")


def parse_config(path: str, dataclass_name: type[T]) -> T:
    config = read_to_dict(path)
    # rich.print(config)
    return dacite.from_dict(data_class=dataclass_name, data=config)
