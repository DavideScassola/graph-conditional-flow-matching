from dataclasses import dataclass
from pathlib import Path

from src.data import DatasetConfig
from src.util import (get_available_device, load_module, pickle_load,
                      pickle_store)

from .models.model import Model

EXPERIMENT_FILE_NAME = "experiment.pkl"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: Model
    generation_options: dict
    name: str
    device: str = get_available_device()
    seed: int | None = None
    train: bool = True
    store: bool = False


def getConfig(path: str | Path) -> ExperimentConfig:
    return load_module(str(path)).CONFIG


def get_experiment_file(path: str) -> str:
    return path + f"/{EXPERIMENT_FILE_NAME}"


def load_experiment(path: str | Path) -> ExperimentConfig:
    return pickle_load(get_experiment_file(str(path)))


def store_experiment(config: ExperimentConfig, path: str | Path) -> None:
    pickle_store(config, file=get_experiment_file(str(path)))
