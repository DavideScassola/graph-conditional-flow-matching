import os
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

FILE_NAME = "data/heart_disease.csv"
SHUFFLE_SEED = 510


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    heart_disease = fetch_ucirepo(id=45)

    if heart_disease.data is None:
        raise ValueError("Dataset not found")

    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)

    os.makedirs("data", exist_ok=True)

    df.sample(frac=1, random_state=SHUFFLE_SEED).to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
