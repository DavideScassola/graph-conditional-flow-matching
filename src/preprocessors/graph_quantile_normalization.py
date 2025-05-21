from torch_geometric.data import HeteroData

from .preprocessor import Preprocessor
from .quantile_normalizer import QuantileNormalizer


class GraphQuantilesNormalizer(Preprocessor):

    def __init__(self):
        self.parameters = {}
        self.transformers = {}

    def fit(self, hetero_graph: HeteroData):
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = QuantileNormalizer()
                p.fit(hetero_graph[table_name].x_continuous)
                self.transformers[table_name] = p
        return self.parameters

    def transform(self, hetero_graph: HeteroData) -> HeteroData:
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = self.transformers[table_name]
                hetero_graph[table_name].x_continuous = p.transform(
                    hetero_graph[table_name].x_continuous
                )
        return hetero_graph

    def reverse_transform(self, hetero_graph: HeteroData) -> HeteroData:
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = self.transformers[table_name]
                hetero_graph[table_name].x_continuous = p.reverse_transform(
                    hetero_graph[table_name].x_continuous
                )
        return hetero_graph

    def serialize(self, p: dict):
        pass

    def deserialize(self, p: dict):
        pass

    def store(self, model_path: str, tag: str = ""):
        for table_name, p in self.transformers.items():
            p.store(model_path, tag=table_name)
