import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from src.preprocessors.quantizer import digits2int_series, int_series2digits
from src.util import pickle_load, pickle_store

from .preprocessor import Preprocessor


class DigitsTransform(Preprocessor):

    def __init__(
        self,
        quantile_transformation: str | None = None,
        include_floats: bool = True,
        include_ints: bool = True,
        digits: int = 4,
        base: int = 10,
    ):
        super().__init__()
        self.quantile_transformation = quantile_transformation
        self.include_floats = include_floats
        self.include_ints = include_ints
        self.digits = digits
        self.base = base

    def fit(self, x: pd.DataFrame):
        self.parameters["columns"] = x.columns
        self.parameters["target_columns"] = {}
        for c in x.columns:
            if (self.include_floats and x[c].dtype == np.float64) or (
                self.include_ints and x[c].dtype == np.integer
            ):
                self.parameters["target_columns"][c] = {"type": x[c].dtype}
                if self.quantile_transformation is not None:
                    transformer = QuantileTransformer(
                        output_distribution=self.quantile_transformation
                    )
                else:
                    transformer = MinMaxScaler(feature_range=(0, 1 - 1e-6))

                self.parameters["target_columns"][c]["transformer"] = transformer.fit(
                    x[c].to_numpy().reshape(-1, 1)
                )

        return self.parameters

    def transform(self, x: pd.DataFrame):
        df = x  # .copy()
        for c in self.parameters["target_columns"]:
            df[c] = self.parameters["target_columns"][c]["transformer"].transform(
                df[c].to_numpy().reshape(-1, 1)
            )
            digits = int_series2digits(
                np.round(df[c] * (self.base**self.digits - 1)).astype(int),
                base=self.base,
            )
            for i in range(digits.shape[1]):
                df[f"{c}_digit_{i}"] = digits[:, i]
                df[f"{c}_digit_{i}"] = df[f"{c}_digit_{i}"].astype("category")
            del df[c]
        return df

    def reverse_transform(self, x: pd.DataFrame):
        df = x  # .copy()
        for c in self.parameters["target_columns"]:
            digits = []
            for i in range(self.digits):
                digit_col = f"{c}_digit_{i}"
                digits.append(df[digit_col].astype(int).values)
                del df[digit_col]

            digits = torch.stack([torch.tensor(d) for d in digits], dim=1)
            df[c] = digits2int_series(digits, base=self.base) / (
                self.base**self.digits - 1
            )
            df[c] = (
                self.parameters["target_columns"][c]["transformer"]
                .inverse_transform(df[c].values.reshape(-1, 1))
                .flatten()
            )
            if self.parameters["target_columns"][c]["type"] == int:
                df[c] = df[c].round()
                df[c] = df[c].astype(int)
            if self.parameters["target_columns"][c]["type"] == float:
                # add some uniform noise for less significant digits
                noise = np.random.uniform(
                    0, 1 / self.base**self.digits, size=df[c].shape[0]
                )
                df[c] += noise

        return df[self.parameters["columns"]]

    def load_(self, model_path: str, tag: str = ""):
        self.parameters = pickle_load(
            str(self.parameters_file(model_path, tag=tag, extension=".pkl"))
        )

    def store(self, model_path: str, tag: str = ""):
        pickle_store(
            self.parameters,
            file=str(self.parameters_file(model_path, tag=tag, extension=".pkl")),
        )
