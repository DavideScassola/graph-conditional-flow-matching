import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
FILE_NAME = "data/household_power_consumption.txt"


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def zip_path() -> str:
    return f"data/{self_base_name()}.zip"


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def extract_zip(archive: str) -> None:
    with zipfile.ZipFile(file=archive, mode="r") as zip_ref:
        zip_ref.extractall(path=Path(archive).parent)
    os.remove(path=archive)


def main():
    os.makedirs(name="data", exist_ok=True)
    if not os.path.isfile(csv_path()):
        urllib.request.urlretrieve(url=ZIP_URL, filename=zip_path())
        extract_zip(archive=zip_path())
        os.rename(src=FILE_NAME, dst=csv_path())
    else:
        print(f"{csv_path()} already present")

    df = pd.read_csv(
        csv_path(),
        sep=";",
        parse_dates={"dt": ["Date", "Time"]},
        dayfirst=True,
        low_memory=False,
        na_values=["nan", "?"],
        index_col="dt",
    )

    df = df.resample("H").median()  # first ?
    df = df[
        "2006-12-17":"2010-11-25"
    ]  # .dropna() # discard first and last day since they are not complete
    df_grouped = df.groupby(df.index.to_period("D"))
    X = np.stack(df_grouped.apply(np.array))
    X = X[~np.any(np.isnan(X), axis=(1, 2))]


if __name__ == "__main__":
    main()
