import json
import tomllib
from typing import Any


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)
