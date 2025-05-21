import numpy as np
import pandas as pd

from src.graph_data import MultiTableDataset
from src.util import fake_float_to_int

from .preprocessor import Preprocessor


def fillna_for_numerical(s: pd.Series, strategy: str = "resample") -> pd.Series:

    if strategy == "resample":
        fill_value = s.mean()
        is_na = s.isna()
        return pd.Series(np.where(is_na, np.random.choice(s[~is_na], size=len(s)), s))

    if strategy == "mean":
        fill_value = s.mean()
        is_na = s.isna()
        if (s[~is_na] - s[~is_na].astype(int) == 0).all():
            fill_value = round(fill_value)  # TODO: we could also use 0
        return s.fillna(fill_value)


def nan_handling(table: pd.DataFrame, missing_token: str) -> None:
    for column in table.columns:
        if table[column].isna().any():
            if table[column].dtype in [object, str, bool, "category"]:
                table[column] = table[column].astype("str")
                table[column] = (
                    table[column].fillna(missing_token).replace("nan", missing_token)
                )
                table[column] = table[column].astype("category")

            elif table[column].dtype in [int, float]:
                table[f"{column}_is_missing"] = table[column].isna()
                table[column] = fillna_for_numerical(table[column], strategy="resample")

    # Pandas doesn't support NaN for integer columns, so we need to convert them from float to int
    fake_float_to_int(table)


def nan_handling_reverse(
    table: pd.DataFrame, missing_token: str, fake_float_columns
) -> None:

    for column in fake_float_columns:
        table[column] = table[column].astype(float).round()

    for column in table.columns:

        if column[-len("_is_missing") :] == "_is_missing":
            prefix = column[: -len("_is_missing")]
            table[prefix] = table[prefix].mask(table[column], pd.NA)
            table.drop(column, axis=1, inplace=True)

        elif table[column].dtype in [object, str, bool, "category"]:
            table[column] = table[column].astype(str).replace(missing_token, pd.NA)


def find_fake_float_columns(table: pd.DataFrame) -> list:
    fake_float_columns = []
    for column in table.columns:
        subset = table[column].dropna()
        if table[column].dtype == float and (subset - subset.astype(int) == 0).all():
            fake_float_columns.append(column)
    return fake_float_columns


class MultiTableMissingValuesHandler(Preprocessor):

    MISSING_TOKEN = "[MISSING]"

    def fit(self, mtd: MultiTableDataset):
        for name in mtd.names():
            self.parameters[name] = {}
            self.parameters[name]["fake_float_columns"] = find_fake_float_columns(
                mtd.features[name]
            )
        return self.parameters

    def transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        for name in mtd.names():
            nan_handling(mtd.features[name], self.MISSING_TOKEN)
        return mtd

    def reverse_transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        for name in mtd.names():
            nan_handling_reverse(
                mtd.features[name],
                self.MISSING_TOKEN,
                fake_float_columns=self.parameters[name]["fake_float_columns"],
            )
        return mtd
