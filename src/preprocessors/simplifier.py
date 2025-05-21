import hashlib

import pandas as pd

from src.graph_data import MultiTableDataset

from .preprocessor import Preprocessor


def find_duplicate_columns(df: pd.DataFrame) -> dict:
    """
    Finds columns that are duplicates of other columns but with different names.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary where keys are the kept column names and values are lists of removed column names.
    """
    # Create a dictionary to store the hash of each column
    column_hashes = {}
    removed_columns_dict = {}

    for col in df.columns:
        # Compute a hash for the column
        col_hash = hashlib.md5(
            pd.util.hash_pandas_object(df[col], index=False).values
        ).hexdigest()

        if col_hash in column_hashes:
            kept_col = column_hashes[col_hash]
            if kept_col not in removed_columns_dict:
                removed_columns_dict[kept_col] = []
            removed_columns_dict[kept_col].append(col)
        else:
            column_hashes[col_hash] = col

    return removed_columns_dict


def remove_duplicate_columns(
    df: pd.DataFrame, removed_columns_dict: dict
) -> pd.DataFrame:
    """
    Removes duplicate columns based on the dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    removed_columns_dict (dict): The dictionary of removed columns.

    Returns:
    pd.DataFrame: The DataFrame with duplicate columns removed.
    """
    columns_to_remove = [col for cols in removed_columns_dict.values() for col in cols]
    df = df.drop(columns=columns_to_remove)
    return df


def reintroduce_duplicate_columns(
    df: pd.DataFrame, removed_columns_dict: dict
) -> pd.DataFrame:
    """
    Reintroduces duplicate columns based on the dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    removed_columns_dict (dict): The dictionary of removed columns.

    Returns:
    pd.DataFrame: The DataFrame with duplicate columns reintroduced.
    """
    for kept_col, removed_cols in removed_columns_dict.items():
        for col in removed_cols:
            df[col] = df[kept_col]
    return df


class Simplifier(Preprocessor):

    def fit(self, mtd: MultiTableDataset):
        for name, df in mtd.features.items():
            d = find_duplicate_columns(df)
            self.parameters[name] = d
        return self.parameters

    def transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        for name, df in mtd.features.items():
            remove_duplicate_columns(df, self.parameters[name])
        return mtd

    def reverse_transform(self, mtd: MultiTableDataset) -> MultiTableDataset:
        for name, df in mtd.features.items():
            reintroduce_duplicate_columns(df, self.parameters[name])
        return mtd
