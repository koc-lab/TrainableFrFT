from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


@dataclass
class TrainerConfig:
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    momentum: float = 0.9
    device: str = "cpu"


@dataclass
class VGGConfig:
    model_name: str = "VGG11"
    n_class: int = 10


@dataclass
class SweepConfig:
    model_name: dict[str, list[str]]
    learning_rate: dict[str, Union[str, float]]
    batch_size: dict[str, Union[str, float]]

    epochs: dict[str, int]
    classes: dict[str, int]
    dataset: dict[str, str]
    device: dict[str, str]

    weight_decay: dict[str, float]
    momentum: dict[str, float]


class PoolType(Enum):
    MaxPool = auto()
    FrFTPool = auto()
    DFrFTPool = auto()
    FFTPool = auto()
