from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.nn.node_conditional_denoiser import (NodeConditionalDenoiser,
                                              NodeConditionalDenoiserConfig)
from src.nn.node_embedder import NodeEmbedderConfig
from src.util import get_hetero_data_device

from .module_config import ModuleConfig


class GraphDenoisingNet(torch.nn.Module):
    def __init__(
        self,
        train_graph: HeteroData,
        node_embedder_config: NodeEmbedderConfig | None,
        node_conditional_denoiser_config: NodeConditionalDenoiserConfig,
        conditioning_tables: list[str] = [],
    ) -> None:
        super().__init__()
        self.actual_num_classes = None
        self.marginal_logits = None
        self.node_embedder = (
            node_embedder_config.build(hetero_graph=train_graph)
            if (node_embedder_config is not None)
            and (len(train_graph.metadata()[1]) > 0)
            and node_embedder_config.embedding_dim > 0
            else None
        )
        self.conditioning_tables = conditioning_tables
        self.node_conditional_denoiser: NodeConditionalDenoiser = (
            node_conditional_denoiser_config.build(
                hetero_graph=train_graph,
                node_embedding_dims=(
                    self.node_embedder.embedding_dims
                    if self.node_embedder is not None
                    else None
                ),
            )
        )
        self.to(get_hetero_data_device(train_graph))

    def predict_x_clean(
        self, *, x_t: HeteroData, t: Tensor, softmax_discrete_features: bool = False
    ) -> HeteroData:
        if self.node_embedder is not None:
            self.node_embedder.add_node_embeddings(input_graph=x_t, t=t)

        x_clean_prediction = self.node_conditional_denoiser.denoise(
            x_t_with_node_embeddings=x_t,
            t=t,
            softmax_discrete_features=softmax_discrete_features,
        )
        return x_clean_prediction


@dataclass
class GraphDenoisingNetConfig(ModuleConfig):
    node_embedder_config: NodeEmbedderConfig | None
    node_conditional_denoiser_config: NodeConditionalDenoiserConfig
    conditioning_tables: list[str] | None = None

    def build_fit(self, train_graph: HeteroData) -> GraphDenoisingNet:
        return GraphDenoisingNet(
            train_graph=train_graph,
            node_embedder_config=self.node_embedder_config,
            node_conditional_denoiser_config=self.node_conditional_denoiser_config,
            conditioning_tables=(
                self.conditioning_tables if self.conditioning_tables is not None else []
            ),
        )
