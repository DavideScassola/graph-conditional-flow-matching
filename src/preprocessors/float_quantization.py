import pandas as pd

from .preprocessor import Preprocessor


def infer_precision(x: pd.Series):
    str_vals = x.astype(str).str.split(".")
    decimal_parts = str_vals.str[1].fillna("")
    decimal_lengths = decimal_parts.str.len()
    return int(decimal_lengths.max())


class FloatQuantization(Preprocessor):

    def fit(self, x: pd.DataFrame):
        self.parameters["precision"] = {
            column: infer_precision(x[column])
            for column in x.columns
            if x[column].dtype == float
        }
        return self.parameters

    def transform(self, x: pd.DataFrame):
        return x

    def reverse_transform(self, x: pd.DataFrame):
        df = x.copy()
        for c, precision in self.parameters["precision"].items():
            df[c] = df[c].round(precision)
        return df
