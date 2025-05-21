from dataclasses import dataclass
from typing import Callable, List

import torch
from torch import Tensor, nn

from src.nn.module_config import ModuleConfig

RESIDUAL_NORMALIZATION_CONSTANT = 1  # / math.sqrt(2)


@dataclass
class MLPconfig(ModuleConfig):
    hidden_channels: List[int]
    activation_layer: Callable = torch.nn.ReLU
    dropout: float = 0.0
    batch_norm: bool = False

    def get_module_class(self):
        return TimeResidualMLP


class FlexibleResidualBlock(nn.Module):
    """
    This block is a simple layer of an MLP if the input and output dimensions are different,
    otherwise it's a residual block, where the input is summed to the output
    """

    def __init__(
        self, *, input_size: int, output_size: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

        self.f = (
            self.residual_layer if input_size == output_size else self.standard_layer
        )

    def residual_layer(self, x: Tensor) -> Tensor:
        return (x + self.activation(self.linear(x))) * RESIDUAL_NORMALIZATION_CONSTANT

    def standard_layer(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class TimeResidualMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        activation_layer: nn.Module,
        batch_norm: bool = False,
        layer_norm: bool = False,  # Add layer_norm option
        dropout: float = 0.0,
        time_embedding: Callable | None = None
    ):
        super().__init__()
        self.time_embedding = time_embedding
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout
        t_emb_dim = (
            self.time_embedding(torch.tensor([[1]])).shape[-1]
            if self.time_embedding
            else 0
        )

        concat_size = in_channels + t_emb_dim
        layers = []
        layers.append(
            FlexibleResidualBlock(
                input_size=concat_size,
                output_size=hidden_channels[0],
                activation=activation_layer,
            )
        )
        if self.dropout > 0.0:
            layers.append(torch.nn.Dropout(p=self.dropout))

        for i in range(len(hidden_channels) - 1):
            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(num_features=hidden_channels[i]))
            elif self.layer_norm:
                layers.append(torch.nn.LayerNorm(normalized_shape=hidden_channels[i]))
            layers.append(
                FlexibleResidualBlock(
                    input_size=hidden_channels[i],
                    output_size=hidden_channels[i + 1],
                    activation=activation_layer,
                )
            )
            if self.dropout > 0.0:
                layers.append(torch.nn.Dropout(p=self.dropout))

        layers.append(
            torch.nn.Linear(
                in_features=hidden_channels[-1],
                out_features=out_channels,
                bias=True,  # TODO: this could be unnecessary
            )
        )
        self.sequential = torch.nn.Sequential(*layers)

    def forward(
        self, X: torch.Tensor, t: torch.Tensor | None = None, keep_shape=False
    ) -> torch.Tensor:
        if self.time_embedding is not None:
            t_emb = self.time_embedding(t)
            if t_emb.shape[0] == 1:
                t_emb = t_emb.repeat(X.shape[0], 1)
            X = torch.cat((X, t_emb), dim=-1)
        out = self.sequential(X.flatten(1))
        return out if not keep_shape else out.reshape(X.shape)
