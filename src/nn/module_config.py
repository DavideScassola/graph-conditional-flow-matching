from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from src.util import load_json, store_json


@dataclass
class ModuleConfig:
    def build_fit(self, data_samples: Tensor):
        return self.get_module_class()(
            input_shape=data_samples[0].shape,
            device=data_samples.device,
            **asdict(self)
        )

    def build(
        self,
        *,
        input_shape: int | list[int],
        output_shape: int | list[int],
        device: torch.device | str
    ) -> nn.Module:
        return self.get_module_class()(
            in_channels=input_shape, out_channels=output_shape, **asdict(self)
        ).to(device)

    @abstractmethod
    def get_module_class(self) -> Callable:
        pass

    def store(self, file: str):
        store_json(asdict(self), file=file)


def load_module_config(file: str) -> ModuleConfig:
    config = load_json(file)
    return ModuleConfig(**config)
