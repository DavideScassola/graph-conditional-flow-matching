from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.graph_data import dummy_tables
from src.nn.mlp import TimeResidualMLP
from src.nn.module_config import ModuleConfig
# from src.nn.tab_ddpm.tab_ddpm.modules import MLPDiffusion
from src.util import log_marginals


def flattened_logits_to_probabilities(logits: Tensor, slices: list):
    return torch.cat(
        [torch.softmax(logits[:, s[0] : s[1]], dim=-1) for s in slices], dim=-1
    )


class NodeConditionalDenoiser(torch.nn.Module):
    def __init__(
        self,
        node_embedding_dims: dict | None,
        hetero_graph: HeteroData,
        conditioning_tables: list[str] = [],
    ) -> None:
        super().__init__()
        self.node_embedding_dims = node_embedding_dims
        self.conditioning_tables = conditioning_tables
        self.continuous_feature_shapes = {
            name: feature.shape[1:]
            for name, feature in hetero_graph.x_continuous_dict.items()
        }
        self.discrete_feature_shapes = {
            name: feature.shape[1:]
            for name, feature in hetero_graph.x_discrete_dict.items()
        }

        self.conditioning_tables += dummy_tables(hetero_graph)

        self.discrete_features_slices = getattr(
            hetero_graph, "discrete_feature_slices", None
        )
        self.fit_log_marginals(hetero_graph)

    def fit_log_marginals(self, hetero_graph: HeteroData) -> None:
        if self.discrete_features_slices is None:
            self.log_marginals = {
                feature_name: (
                    log_marginals(one_hot_data=hetero_graph[feature_name].x_discrete)
                    if hetero_graph[feature_name].x_discrete.numel() > 0
                    else torch.tensor(0.0)
                )
                for feature_name in hetero_graph.metadata()[0]
                if feature_name not in self.conditioning_tables
            }

            # Register log_marginals as buffers
            for name, tensor in self.log_marginals.items():
                self.register_buffer(f"log_marginal_{name}", tensor)
        else:
            self.log_marginals = {}
            for feature_name, slices in self.discrete_features_slices.items():
                self.log_marginals[feature_name] = (
                    torch.cat(
                        [
                            log_marginals(
                                one_hot_data=hetero_graph[feature_name].x_discrete[
                                    :, s[0] : s[1]
                                ]
                            )
                            for s in slices
                        ],
                        dim=-1,
                    )
                    if hetero_graph[feature_name].x_discrete.numel() > 0
                    else torch.tensor(0.0)
                )

    def get_feature_shapes(self):
        return self.feature_shapes

    def get_embedding_dim(self, name: str):
        return (
            self.node_embedding_dims[name]
            if self.node_embedding_dims is not None
            else 0
        )

    @abstractmethod
    def feature_denoise(
        self,
        *,
        hetero_graph_with_node_embeddings: HeteroData,
        feature_name: str,
        t: Tensor | None,
    ) -> Dict[str, Tensor]:
        pass

    def to_discrete_probabilities(self, logits: Tensor, feature_name: str):
        if self.discrete_features_slices is not None:
            return flattened_logits_to_probabilities(
                logits, slices=self.discrete_features_slices[feature_name]
            )
        return torch.softmax(logits, dim=-1)

    def denoise(
        self,
        x_t_with_node_embeddings: HeteroData,
        t: Tensor,
        softmax_discrete_features: bool = False,
    ) -> HeteroData:
        predicted_graph = x_t_with_node_embeddings.clone()

        for feature_name in x_t_with_node_embeddings.x_continuous_dict.keys():
            if feature_name in self.conditioning_tables:
                del predicted_graph[feature_name]
            else:
                predicted_feature = self.feature_denoise(
                    hetero_graph_with_node_embeddings=predicted_graph,
                    t=t,
                    feature_name=feature_name,
                )

                if predicted_graph[feature_name].x_discrete.numel() > 0:
                    predicted_graph[feature_name].x_discrete = (
                        predicted_feature["discrete"]
                        if not softmax_discrete_features
                        else self.to_discrete_probabilities(
                            predicted_feature["discrete"], feature_name
                        )
                    )

                predicted_graph[feature_name].x_continuous = predicted_feature[
                    "continuous"
                ]
        return predicted_graph


@dataclass
class NodeConditionalDenoiserConfig(ModuleConfig):

    @abstractmethod
    def build(
        self, *, hetero_graph: HeteroData, node_embedding_dims: int
    ) -> NodeConditionalDenoiser:
        pass


class NodeConditionalMLP_Denoiser(NodeConditionalDenoiser):

    def __init__(
        self,
        node_embedding_dims: dict,
        hetero_graph: HeteroData,
        hidden_channels: tuple | dict,
        dropout: float | dict,
        residual_io: bool = False,
        **mlp_args,
    ) -> None:
        self.residual_io = residual_io
        super().__init__(node_embedding_dims, hetero_graph)

        if not isinstance(hidden_channels, dict):
            hidden_channels = {
                feature_name: hidden_channels
                for feature_name in hetero_graph.metadata()[0]
                if feature_name not in self.conditioning_tables
            }

        if not isinstance(dropout, dict):
            dropout = {
                feature_name: dropout
                for feature_name in hetero_graph.metadata()[0]
                if feature_name not in self.conditioning_tables
            }

        self.models = torch.nn.ModuleDict(
            {
                feature_name: self.build_mlp(
                    feature_name,
                    hetero_graph,
                    hidden_channels=hidden_channels[feature_name],
                    dropout=dropout[feature_name],
                    **mlp_args,
                )
                for feature_name in hetero_graph.metadata()[0]
                if feature_name not in self.conditioning_tables
            }
        )

    def build_mlp(
        self,
        feature_name: str,
        hetero_graph: HeteroData,
        **mlp_args,
    ) -> torch.nn.Module:

        flattened_discrete_features_size = (
            int(np.prod(np.array(self.discrete_feature_shapes[feature_name])))
            if feature_name in self.discrete_feature_shapes
            else 0
        )

        flattened_continuous_features_size = (
            int(np.prod(np.array(self.continuous_feature_shapes[feature_name])))
            if feature_name in self.continuous_feature_shapes
            else 0
        )

        return TimeResidualMLP(
            in_channels=flattened_discrete_features_size
            + flattened_continuous_features_size
            + self.get_embedding_dim(feature_name),
            out_channels=flattened_discrete_features_size
            + flattened_continuous_features_size,
            **mlp_args,
        )

    def nn_input(
        self,
        hetero_graph_with_node_embeddings: HeteroData,
        feature_name: str,
    ) -> Tensor:

        input_features = [
            hetero_graph_with_node_embeddings[feature_name].x_continuous.flatten(1),
            hetero_graph_with_node_embeddings[feature_name].x_discrete.flatten(1),
        ]

        if hasattr(hetero_graph_with_node_embeddings[feature_name], "node_embeddings"):
            input_features += [
                hetero_graph_with_node_embeddings[feature_name].node_embeddings
            ]

        return torch.concat(input_features, dim=1)

    def feature_denoise(
        self,
        *,
        hetero_graph_with_node_embeddings: HeteroData,
        feature_name: str,
        t: Tensor,
    ) -> Dict[str, Tensor]:

        nn_input = self.nn_input(
            hetero_graph_with_node_embeddings=hetero_graph_with_node_embeddings,
            feature_name=feature_name,
        )

        flattened_discrete_features_size = np.prod(
            np.array(self.discrete_feature_shapes[feature_name])
        )
        out = self.models[feature_name](X=nn_input, t=t)
        discrete_out = out[:, :flattened_discrete_features_size].reshape(
            out.shape[0], *self.discrete_feature_shapes[feature_name]
        ) + self.log_marginals[feature_name].to(
            out.device
        )  # TODO: not sure if to(device) is expensive
        continuous_out = out[:, flattened_discrete_features_size:].reshape(
            out.shape[0], *self.continuous_feature_shapes[feature_name]
        )
        if self.residual_io:
            continuous_out.add_(
                hetero_graph_with_node_embeddings[feature_name].x_continuous
            )

        return {"discrete": discrete_out, "continuous": continuous_out}


@dataclass
class NodeConditionalMLP_DenoiserConfig(NodeConditionalDenoiserConfig):
    hidden_channels: tuple | dict
    activation_layer: torch.nn.Module
    dropout: float | dict = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    time_embedding: Callable | None = None
    residual_io: bool = False

    def build(
        self, *, hetero_graph: HeteroData, node_embedding_dims: dict
    ) -> NodeConditionalDenoiser:
        return NodeConditionalMLP_Denoiser(
            node_embedding_dims=node_embedding_dims,
            hetero_graph=hetero_graph,
            **vars(self),
        )
