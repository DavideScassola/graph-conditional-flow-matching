import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix

from src.data import DatasetConfig
from src.models.model import Model
from src.util import get_hetero_data_device

MAX_COMPONENT_SIZE_THRESHOLD = 0.5


def get_connected_components(data, skip_threshold=MAX_COMPONENT_SIZE_THRESHOLD):
    """
    Code inspired from https://github.com/ValterH/relational-graph-conditioned-diffusion (2024) for reproducibility purposes
    """

    device = get_hetero_data_device(data)

    names = data.metadata()[0]
    for name in names:
        data[name]["num_nodes"] = len(data[name]["x_discrete"])

    homo = data.to_homogeneous()

    # for name in names:
    #    del data[name]['num_nodes']

    adj = to_scipy_sparse_matrix(homo.edge_index.cpu())

    num_components, component = sp.csgraph.connected_components(adj, connection="weak")
    components = dict()
    for i, key in enumerate(names):
        components[key] = component[homo.node_type.cpu() == i]

    cc_sizes = np.bincount(component)
    max_cc_size = cc_sizes.max()

    if max_cc_size / data.num_nodes > skip_threshold:
        return [data], max_cc_size

    connected_components = []

    for component in np.arange(num_components):
        nodes = dict()
        for key, ccs in components.items():
            nodes[key] = torch.tensor(ccs == component, device=device)
        subgraph = data.subgraph(nodes)
        connected_components.append(subgraph)

    return connected_components, max_cc_size


def sample_connected_components(train_hetero_data):
    """
    Code inspired from https://github.com/ValterH/relational-graph-conditioned-diffusion (2024) for reproducibility purposes
    """

    subgraphs, num_nodes_largest = get_connected_components(train_hetero_data)

    pct_largest = num_nodes_largest / train_hetero_data.num_nodes
    if pct_largest > MAX_COMPONENT_SIZE_THRESHOLD:
        print(
            "\033[93m"
            + f"The largest connected component is larger than {MAX_COMPONENT_SIZE_THRESHOLD: .1%} of the dataset ({pct_largest * 100: .2f}%), the original graph will be used."
            + "\x1b[0m"
        )
        return train_hetero_data

    num_structures = len(subgraphs)

    if num_structures >= len(subgraphs):
        samples = random.choices(subgraphs, k=num_structures)
    else:
        samples = random.sample(subgraphs, k=num_structures)

    # reporder the samples, s.t. the last sample has all tables (this will create a valid edge_index)
    for i in range(len(samples)):
        if samples[i].metadata() == train_hetero_data.metadata():
            # move the current subgraph to the end of the list
            samples.append(samples.pop(i))
            break

    # use the dataloader to stitch the samples to a single HeteroData object
    dataloader = DataLoader(samples, batch_size=num_structures)
    hetero_data = next(iter(dataloader))

    for k in hetero_data.metadata()[0]:
        del hetero_data[k]["ptr"]
        del hetero_data[k]["batch"]

    return hetero_data


def delete_features_content(data: HeteroData):
    for key in data.metadata()[0]:
        data[key].x_discrete *= 0
        data[key].x_continuous *= 0


class ResampleConnectedComponents(Model):
    """
    This "model" simply copies the graph of given training data.
    """

    MODEL_FOLDER_NAME = Path("graph_model")

    def __init__(self, keep_original_graph: bool = False) -> None:
        super().__init__()
        self.keep_original_graph = keep_original_graph

    def train(self, train_graph: HeteroData):
        self.train_graph = train_graph.clone()
        # delete_features_content(self.train_graph)
        self.discrete_feature_slices = train_graph.discrete_feature_slices
        del self.train_graph.discrete_feature_slices

    def generate(self):
        if self.keep_original_graph:
            graph = self.train_graph
        else:
            graph = sample_connected_components(self.train_graph)
        graph.discrete_feature_slices = self.discrete_feature_slices
        return graph

    def store(self, experiment_path: str):
        graph_model_path = self.get_model_folder(experiment_path)
        os.makedirs(graph_model_path)
        # TODO: implement

    def load_(self, experiment_path: str):
        self.get_model_folder(experiment_path)
        # TODO: implement
        # self.graph = nx.read_graphml(graph_model_path)

    def _train(self, X):
        pass

    def _generate(self, n_samples: int):
        pass

    def _store(self, model_path):
        pass

    def _load_(self, model_path):
        pass

    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: DatasetConfig,
        generation_options: dict,
        **kwargs,
    ):
        pass
