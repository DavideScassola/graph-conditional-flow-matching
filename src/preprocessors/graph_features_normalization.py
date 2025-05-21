from torch_geometric.data import HeteroData

from .mean_std_normalizer import MeanStdNormalizer
from .preprocessor import Preprocessor


class GraphFeaturesNormalizer(Preprocessor):

    def fit(self, hetero_graph: HeteroData):
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = MeanStdNormalizer()
                p.fit(hetero_graph[table_name].x_continuous)
                self.parameters[table_name] = p.parameters
        return self.parameters

    def transform(self, hetero_graph: HeteroData) -> HeteroData:
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = MeanStdNormalizer()
                p.parameters = self.parameters[table_name]
                hetero_graph[table_name].x_continuous = p.transform(
                    hetero_graph[table_name].x_continuous
                )
        return hetero_graph

    def reverse_transform(self, hetero_graph: HeteroData) -> HeteroData:
        for table_name in hetero_graph.metadata()[0]:
            if hetero_graph[table_name].x_continuous.numel() > 0:
                p = MeanStdNormalizer()
                p.parameters = self.parameters[table_name]
                hetero_graph[table_name].x_continuous = p.reverse_transform(
                    hetero_graph[table_name].x_continuous
                )
        return hetero_graph

    def serialize(self, p: dict):
        return {
            name: MeanStdNormalizer().serialize(parameter)
            for name, parameter in p.items()
        }

    def deserialize(self, p: dict):
        return {k: MeanStdNormalizer().deserialize(v) for k, v in p.items()}
