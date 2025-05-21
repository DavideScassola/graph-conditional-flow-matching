import os
import sys
from pathlib import Path

sys.path.append(".")
from ucimlrepo import fetch_ucirepo

SHUFFLE_SEED = 510

INTERESTING_COLUMNS = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income',
    ]

def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    # fetch dataset
    print("Fetching dataset...")
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    df = adult.data.original
    os.makedirs("data", exist_ok=True)
    #df = df.dropna()
    #df.sample(frac=1, random_state=SHUFFLE_SEED).to_csv(csv_path(), index=False)
    
    df['income'].str.replace('.', '')
    df[INTERESTING_COLUMNS].to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
