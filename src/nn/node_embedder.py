from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GIN, GATv2Conv, Linear, to_hetero

from src.graph_data import get_neighbours_count_tables, get_node_degree_stats
from src.nn.module_config import ModuleConfig
from src.util import get_hetero_data_device


class NodeEmbedder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        gnn: torch.nn.Module,
        hetero_graph: HeteroData,
        add_node_degrees: bool = True,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.add_node_degrees = add_node_degrees
        self.node_degree_stats = None
        self.cached_node_degrees = None
        self.embedding_dims = self.initialize_node_embedding_dims(
            hetero_graph=hetero_graph, node_embedding_dim=embedding_dim
        )

    def initialize_node_embedding_dims(
        self, *, node_embedding_dim: int, hetero_graph: HeteroData
    ) -> dict:
        node_embedding_dims = {
            name: node_embedding_dim for name in hetero_graph.metadata()[0]
        }

        if self.add_node_degrees:
            for name, table in get_neighbours_count_tables(
                hetero_graph
            ).items():  # TODO: inefficient
                node_embedding_dims[name] += table.shape[1]

        return node_embedding_dims

    def get_node_embedding_dims(self) -> dict:
        return self.embedding_dims

    def get_input_dict(self, *, hetero_graph: HeteroData, t):

        d = {
            name: torch.cat(
                [
                    hetero_graph[name][k].flatten(1)
                    for k in hetero_graph[name].keys()
                    if k
                    not in [
                        "edge_index",
                        "subset",
                        "node_embeddings",
                        "num_nodes",
                    ]  # TODO: not ideal
                ],
                # + [t_emb.repeat(hetero_graph[name].x_continuous.size(0), 1)],
                dim=1,
            )
            for name in hetero_graph.metadata()[0]
        }

        # Setting empty tables to zeros, in order to avoid weird behaviours
        for name in d.keys():
            if d[name].numel() == 0:
                d[name] = torch.zeros(d[name].shape[0], 1).to(d[name].device)

        return d

    def update_node_degrees(self, input_graph: HeteroData):
        self.cached_node_degrees = {}
        neighbours_count_tables = get_neighbours_count_tables(
            input_graph, log_count=True
        )
        for name, stats in self.node_degree_stats.items():
            self.cached_node_degrees[name] = (
                neighbours_count_tables[name] - stats["mean"]
            ) / torch.where(
                stats["std"] > 0, stats["std"], torch.ones_like(stats["std"])
            )  # TODO: it would be better to avid 0 std node degrees but then it would mess with the dims

    def update_node_degree_stats(self, input_graph: HeteroData):
        self.node_degree_stats = get_node_degree_stats(input_graph, log_count=True)

    def eval(self):
        super().eval()
        self.cached_node_degrees = None

    def train(self, mode=True):
        super().train(mode)
        self.cached_node_degrees = None
        self.node_degree_stats = None

    def add_node_embeddings(self, input_graph: HeteroData, t):
        """
        The input graph should be an HeteroData object with x_discrete as one_hot encoded features, and x_continuous as continuous features.
        """

        # TODO: dummy_tables to zeros?

        gnn_output = self.gnn(
            self.get_input_dict(hetero_graph=input_graph, t=t),
            input_graph.edge_index_dict,
        )

        # Add node embeddings to the input graph

        if not self.add_node_degrees:
            for name in input_graph.metadata()[0]:
                input_graph[name].node_embeddings = gnn_output[name]
        else:
            if self.node_degree_stats is None:
                self.update_node_degree_stats(input_graph)
            if self.cached_node_degrees is None:
                self.update_node_degrees(input_graph)
            for name in input_graph.metadata()[0]:
                input_graph[name].node_embeddings = (
                    torch.cat((self.cached_node_degrees[name], gnn_output[name]), dim=1)
                    if name in self.cached_node_degrees
                    else gnn_output[name]
                )


@dataclass
class NodeEmbedderConfig(ModuleConfig):
    embedding_dim: int
    add_node_degrees: bool

    @abstractmethod
    def build(self, hetero_graph: HeteroData):
        raise NotImplementedError


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_dim: int) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(
            (-1, -1), hidden_channels, add_self_loops=False, concat=True
        )
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv(
            (-1, -1), embedding_dim, add_self_loops=False, concat=True
        )
        self.lin2 = Linear(-1, embedding_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


@dataclass
class GATConfig(NodeEmbedderConfig):
    hidden_channels: int
    aggregation: str = "sum"

    def build(self, hetero_graph: HeteroData) -> NodeEmbedder:
        model = GAT(
            embedding_dim=self.embedding_dim,
            hidden_channels=self.hidden_channels,
        )

        gnn = to_hetero(model, hetero_graph.metadata(), aggr=self.aggregation)

        return NodeEmbedder(
            embedding_dim=self.embedding_dim,
            gnn=gnn,
            hetero_graph=hetero_graph,
            add_node_degrees=self.add_node_degrees,
        )


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        *,
        gnn: GIN,
        embedder_config=None,
        linear_embedding_size: int | None = None,
        graph: HeteroData,
    ) -> None:
        super().__init__()
        self.embedder_config = embedder_config
        self.linear_embedding_size = linear_embedding_size
        self.hetero_gnn = to_hetero(gnn, graph.metadata(), aggr="sum").to(
            get_hetero_data_device(graph)
        )
        self.embedders = self.initialize_embedders(embedder_config, graph)

    def initialize_embedders(self, embedder_config, graph):
        if embedder_config is None:
            if self.linear_embedding_size is None:
                raise ValueError(
                    "linear_embedding_size must be provided if embedder_config is not provided"
                )
            return torch.nn.ModuleDict(
                {
                    name: Linear(-1, self.linear_embedding_size).to(
                        get_hetero_data_device(graph)
                    )
                    for name in graph.metadata()[0]
                }
            )
        return torch.nn.ModuleDict(
            {
                name: embedder_config.build(graph[name]).to(
                    get_hetero_data_device(graph)
                )
                for name in graph.metadata()[0]
            }
        )

    def embed(self, input_dict):
        return {name: self.embedders[name](input_dict[name]) for name in input_dict}

    def get_input_dict(self, hetero_graph: HeteroData):
        return {
            name: torch.cat(
                [
                    hetero_graph[name][k].flatten(1)
                    for k in hetero_graph[name].keys()
                    if k != "edge_index"
                ],
                dim=1,
            )
            for name in hetero_graph.metadata()[0]
        }

    def hetero_graph_forward(self, hetero_graph: HeteroData):
        return self.hetero_gnn(
            self.embed(self.get_input_dict(hetero_graph)), hetero_graph.edge_index_dict
        )

    def forward(self, input_dict, edge_index_dict):
        return self.hetero_gnn(self.embed(input_dict), edge_index_dict)


class HeteroGinConfig(NodeEmbedderConfig):
    def __init__(
        self,
        *,
        num_layers: int,
        hidden_channels: int,
        embedder_config=None,
        linear_embedding_size: int | None = None,
        embedding_dim: int,
        add_node_degrees: bool,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, add_node_degrees=add_node_degrees)
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.embedder_config = embedder_config
        self.linear_embedding_size = linear_embedding_size

    def build(self, hetero_graph: HeteroData) -> NodeEmbedder:

        gin = GIN(
            in_channels=-1,
            hidden_channels=self.hidden_channels,
            out_channels=self.embedding_dim,
            num_layers=self.num_layers,
        )

        hetero_gnn = HeteroGNN(
            gnn=gin,
            embedder_config=self.embedder_config,
            linear_embedding_size=self.linear_embedding_size,
            graph=hetero_graph,
        )

        return NodeEmbedder(
            embedding_dim=self.embedding_dim,
            gnn=hetero_gnn,
            hetero_graph=hetero_graph,
            add_node_degrees=self.add_node_degrees,
        )
