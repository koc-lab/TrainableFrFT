from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import torchvision.transforms


@dataclass
class DataHandlerConfig:
    batch_size: int
    multi_gpu: bool
    train_slice: int
    test_slice: int
    train_transform: torchvision.transforms
    test_transform: torchvision.transforms


class OptimizerType(Enum):
    # TODO: add other types in future
    SGD = auto()
    Adam = auto()


@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType
    lr: float
    wd: float
    momentum: float


class SchedulerType(Enum):
    CosineAnnealingLR = auto()
    # TODO: add other types in future


@dataclass
class SchedulerConfig:
    scheduler_type: SchedulerType
    max_epochs: int


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
