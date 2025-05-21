from copy import deepcopy
from pathlib import Path

from .preprocessor import Preprocessor


class Compose(Preprocessor):

    def __init__(
        self,
        preprocessors: list[Preprocessor],
    ):
        self.preprocessors = [p for p in preprocessors if p is not None]

        self.parameters = {}

    def _transform(self, x, fit: bool):
        out = deepcopy(x)
        if len(self.preprocessors) == 0:
            return out

        for i in range(0, len(self.preprocessors)):
            if fit:
                self.preprocessors[i].fit(out)
                self.parameters[i] = self.preprocessors[i].parameters
            else:
                self.preprocessors[i].parameters = self.parameters[i]
            out = self.preprocessors[i].transform(out)
        return out

    def fit_transform(self, x):
        return self._transform(x, fit=True)

    def transform(self, x):
        return self._transform(x, fit=False)

    def reverse_transform(self, x):
        out = deepcopy(x)
        if len(self.preprocessors) == 0:
            return out

        for p in self.preprocessors[::-1]:
            out = p.reverse_transform(out)
        return out

    def store(self, model_path: str | Path, tag: str = ""):
        for p in self.preprocessors:
            p.store(model_path, tag=tag)

    def serialize(self, p: dict) -> dict:
        return {
            i: preprocessor.serialize(p[i])
            for i, preprocessor in enumerate(self.preprocessors)
        }

    def deserialize(self, p: dict) -> dict:
        return {
            i: preprocessor.deserialize(p[i])
            for i, preprocessor in enumerate(self.preprocessors)
        }

    def load_(self, model_path: str | Path, tag: str = ""):
        super().load_(model_path, tag)
        for i in range(len(self.preprocessors)):
            self.preprocessors[i].parameters = self.parameters[i]
