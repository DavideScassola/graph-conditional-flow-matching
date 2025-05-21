from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Optimization:
    epochs: int
    batch_size: int
    optimizer_class: Callable
    optimizer_hyperparameters: dict
    gradient_clipping: float | None = None
    parameters_momentum: float | None = None
    interactive: bool = False
    warmup_epochs: int = -1
    grad_accumulation_steps: int = 1
    lr_scheduler_class: Callable | None = None
    lr_scheduler_params: dict | None = None
    patience: int | None = None
    restore_best_validation_model: bool = True

    def build_optimizer(self, params) -> torch.optim.Optimizer:
        return self.optimizer_class(params, **self.optimizer_hyperparameters)

    def build_lr_scheduler(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        if self.lr_scheduler_class is None:
            raise ValueError(
                "lr_scheduler_class and lr_scheduler_params must be set to build a lr_scheduler"
            )
        params = (
            self.lr_scheduler_params if self.lr_scheduler_params is not None else {}
        )
        return self.lr_scheduler_class(optimizer, **params)
