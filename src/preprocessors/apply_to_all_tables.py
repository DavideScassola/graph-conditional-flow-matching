from copy import deepcopy
from pathlib import Path

from src.graph_data import MultiTableDataset

from .preprocessor import Preprocessor


class ApplyToAllTables(Preprocessor):

    def __init__(self, preprocessor: Preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def fit(self, mtd: MultiTableDataset):
        for name in mtd.names():
            self.preprocessor.fit(mtd.features[name])
            self.parameters[name] = deepcopy(self.preprocessor)

    def transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        mtd_out = mtd  # deepcopy(mtd)
        for name in mtd.names():
            mtd_out.features[name] = self.parameters[name].transform(mtd.features[name])
        return mtd_out

    def reverse_transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        mtd_out = mtd  # deepcopy(mtd)
        for name in mtd.names():
            mtd_out.features[name] = self.parameters[name].reverse_transform(
                mtd.features[name]
            )
        return mtd_out

    def store(self, model_path: str | Path, tag: str = ""):
        for name in self.parameters:
            self.parameters[name].store(model_path, tag=name)

    def load_(self, model_path: str | Path, tag: str = ""):
        for name in self.parameters:
            self.parameters[name].load_(model_path, tag=name)
