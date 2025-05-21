from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List

import torch

from src.util import load_json, store_json


class Preprocessor(ABC):
    def __init__(self):
        self.parameters = {}

    def fit(self, x):
        """
        This deafult fit method is a trick to exploit fit_transform, and then restore the original x if it was modified
        """
        x_mod = deepcopy(x)
        self.fit_transform(x_mod)

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def reverse_transform(self, x):
        pass

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def parameters_file(
        self, model_path: str | Path, *, extension: str = ".json", tag: str = ""
    ):
        # TODO: this works only if every kind of preprocessor is used only once
        return Path(model_path) / Path(str(self.__class__.__name__) + tag + extension)

    def load_(self, model_path: str | Path, tag: str = ""):
        self.parameters = self.deserialize(
            load_json(self.parameters_file(model_path, tag=tag))
        )

    def store(self, model_path: str | Path, tag: str = ""):
        store_json(
            self.serialize(self.parameters),
            file=self.parameters_file(model_path, tag=tag),
        )

    def serialize(self, p: dict):
        return p

    def deserialize(self, p: dict):
        return p


class TensorPreprocessor(Preprocessor, ABC):
    def fit(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)


def composed_transform(
    input: torch.Tensor, fit: bool, *, preprocessors: List[TensorPreprocessor]
) -> torch.Tensor:
    if len(preprocessors) == 0:
        return input
    out = (
        preprocessors[0].fit_transform(input)
        if fit
        else preprocessors[0].transform(input)
    )
    for p in preprocessors[1:]:
        out = p.fit_transform(out) if fit else p.transform(out)
    return out


def composed_inverse_transform(
    input: torch.Tensor, *, preprocessors: List[TensorPreprocessor]
) -> torch.Tensor:
    if len(preprocessors) == 0:
        return input
    out = preprocessors[-1].reverse_transform(input)
    for p in preprocessors[-2::-1]:
        out = p.reverse_transform(out)
    return out
