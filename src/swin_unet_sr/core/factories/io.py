from pathlib import Path
from typing import Any, Callable

from ..io import read_json, read_toml


class DuplicateNameError(Exception): ...


# Read configuration from any IO and return a (nested) dictionary
ReadFn = Callable[[str], dict[str, Any]]


_registry: dict[str, ReadFn] = {
    "json": read_json,
    "toml": read_toml,
}


def read_to_dict(path: str) -> dict[str, Any]:
    suffix = Path(path).suffix[1:]  # remove dot
    if suffix not in _registry:
        raise KeyError(f"[{suffix}] is not registered")

    return _registry[suffix](path)
