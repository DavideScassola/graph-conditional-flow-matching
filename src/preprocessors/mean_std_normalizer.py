import torch

from .preprocessor import TensorPreprocessor


class MeanStdNormalizer(TensorPreprocessor):
    def fit(self, x: torch.Tensor):
        self.parameters["mean"] = x.mean(axis=0)  # type: ignore
        self.parameters["std"] = x.std(axis=0)  # type: ignore
        self.parameters["std"][
            self.parameters["std"] == 0.0
        ] = 1.0  # Avoiding 0. division

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self.parameters_to_device(x)
        return (x - self.parameters["mean"]) / self.parameters["std"]

    def parameters_to_device(self, x: torch.Tensor):
        if x.device != self.parameters["mean"].device:
            self.parameters["mean"] = self.parameters["mean"].to(x.device)
            self.parameters["std"] = self.parameters["std"].to(x.device)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.parameters_to_device(x)
        return x * self.parameters["std"] + self.parameters["mean"]

    def serialize(self, p: dict) -> dict:
        return {k: v.tolist() for k, v in p.items()}

    def deserialize(self, p: dict) -> dict:
        return {k: torch.tensor(v) for k, v in p.items()}
