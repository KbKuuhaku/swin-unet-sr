from pathlib import Path

import click
import rich

from .core.config import parse_config


@click.group()
def cli() -> None: ...


@cli.command()
@click.argument("config-name")
def train(config_name: str) -> None:
    from .train import TrainConfig, train_ffhq

    config_path = Path("configs") / "train" / f"{config_name}.toml"
    config = parse_config(path=str(config_path), dataclass_name=TrainConfig)

    rich.print(config)

    train_ffhq(config)


@cli.command()
@click.argument("config-name")
def test(config_name: str) -> None:
    from .test import TestConfig, test_ffhq

    config_path = Path("configs") / "test" / f"{config_name}.toml"
    config = parse_config(path=str(config_path), dataclass_name=TestConfig)

    rich.print(config)

    test_ffhq(config)
