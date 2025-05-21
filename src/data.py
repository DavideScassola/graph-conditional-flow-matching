from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torchvision import datasets, transforms

from src.constants import DATASET_SCRIPTS_FOLDER
from src.util import execute_python_script


def get_subset_slice(dataset, train_prop: float, train: bool = True) -> slice:
    cut = int(len(dataset) * train_prop)
    return slice(None, cut) if train else slice(cut, None)


def generate_from_script(dataset_name: str):
    script = DATASET_SCRIPTS_FOLDER / Path(dataset_name + ".py")
    if not script.is_file():
        raise ValueError(f"Couldn't find script for generating {dataset_name} dataset")
    print(f"Generating dataset from {script} script ...")
    execute_python_script(script)


def generate_if_necessary(path: str | Path) -> None:
    path = Path(path)
    if (not path.is_file()) and (not path.is_dir()):
        print(f"dataset {path} not found")
        generate_from_script(path.stem)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_npy(path: str | Path) -> Tensor:
    return torch.from_numpy(np.load(path)).float()


def loader2tensor(d) -> torch.Tensor:
    # TODO: slow and bad solution, one shouldn't convert back to a single tensor
    return torch.concat([s[0] for s in d])


def get_MNIST(*, train: bool):
    # TODO: transformation is not OK for every model
    print("loading MNIST...")
    return loader2tensor(
        datasets.MNIST(
            root="data",
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
            download=True,
        )
    )


DEFAULT_DATASETS = {"MNIST": get_MNIST}


@dataclass
class DatasetConfig:
    path: str | Path
    train_proportion: float

    def get(self, train=True) -> DataFrame | Tensor:
        if self.path in DEFAULT_DATASETS:
            return DEFAULT_DATASETS[self.path](train=train)
        generate_if_necessary(self.path)
        ext = Path(self.path).suffix[1:]
        data = eval(f"load_{ext}")(self.path)
        s = get_subset_slice(data, self.train_proportion, train=train)
        if isinstance(data, DataFrame):
            return data.iloc[s]
        else:
            return data[s]
