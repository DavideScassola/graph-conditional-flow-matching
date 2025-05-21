import torch
from sklearn.preprocessing import QuantileTransformer

from src.util import pickle_load, pickle_store

from .preprocessor import Preprocessor


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_tensor(x):
    return torch.tensor(x)


class QuantileNormalizer(Preprocessor):
    # TODO: update this, and add non-differentiable transformers like this to model class
    def __init__(self, output_distribution: str = "normal"):
        self.output_distribution = output_distribution

    def fit(self, x):
        n_quantiles = 1000 if x.shape[0] > 1000 else x.shape[0] // 2
        self.parameters = QuantileTransformer(
            output_distribution=self.output_distribution, n_quantiles=n_quantiles
        )
        return self.parameters.fit(to_numpy(x))

    def transform(self, x):
        return to_tensor(self.parameters.transform(to_numpy(x))).to(x.device)

    def reverse_transform(self, x):
        return to_tensor(self.parameters.inverse_transform(to_numpy(x))).to(x.device)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def load_(self, model_path: str, tag: str = ""):
        self.parameters = pickle_load(
            str(self.parameters_file(model_path, tag=tag, extension=".pkl"))
        )

    def store(self, model_path: str, tag: str = ""):
        pickle_store(
            self.parameters,
            file=str(self.parameters_file(model_path, tag=tag, extension=".pkl")),
        )
