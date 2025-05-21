import argparse
import fnmatch
import os
import shutil
import sys

import numpy as np

sys.path.append(".")
import pandas as pd

from src.experiment_config import getConfig
from src.util import load_json, store_json
from syntherela_benchmark.experiments.evaluation.benchmark_lib import \
    run_benchmark

DEFAULT_METHOD_NAME = "ThisPaper"
RESULTS_PATH = "syntherela_benchmark/results"

TARGET_PERFORMANCES = { 'airbnb-simplified_subsampled' : 0.67,
                        'Biodegradability_v1': 0.83,
                        'CORA_v1': 0.60,
                        'imdb_MovieLens_v1': 0.64,
                        'rossmann_subsampled': 0.77,
                        'walmart_subsampled': 0.74}
      
def get_dataset_name(artifact_folder):
    return os.path.basename(getConfig(artifact_folder + "/config.py").dataset.path)


def most_recent_artifact_folder():
    folder_path = "artifacts/models"
    subdirectories = [
        d
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d[0] != "_"
    ]
    subdirectories.sort()
    if subdirectories:
        print("Evaluating", subdirectories[-1])
        return os.path.join(folder_path, subdirectories[-1])
    else:
        return None
    
def build_date_series(df: pd.DataFrame, date_columns: list[str]) -> pd.Series:
    def infer_time_unit(colname: str):
        return colname[colname.rfind('_') + 1:]
    df_temp = pd.DataFrame({infer_time_unit(name): df[name] for name in date_columns})
    return pd.to_datetime(df_temp, errors='coerce')
    
def restore_dates(folder):
    """
    For each .csv file in the folder, restore the original dates, from separate int columns to unified date format
    """
    
    def is_date_column(colname: str):
        has_date_name = ("date" in colname.lower()) or ("timestamp" in colname.lower())
        has_time_unit = colname[colname.rfind('_') + 1:] in ['year', 'month', 'day', 'hour', 'minute', 'second']
        return has_date_name and has_time_unit
    
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(folder + "/" + file)
            all_date_columns = sorted([col for col in df.columns if is_date_column(col)])
            while all_date_columns:
                name = all_date_columns[0][:all_date_columns[0].rfind('_')]
                one_date_columns = list(filter(lambda x: name in x, all_date_columns))
                date = build_date_series(df, one_date_columns)
                df[name] = date
                df.drop(columns=one_date_columns, inplace=True)
                all_date_columns = [col for col in all_date_columns if name not in col]
            df.to_csv(folder + "/" + file, index=False)


def print_performance(my_perf, target_perf, method_name):
    """
    Prints the performance with color coding:
    - Red if `my_perf` > `target_perf`
    - Yellow if `my_perf` == `target_perf`
    - Green if `my_perf` < `target_perf`
    """
    if my_perf > target_perf:
        color = "\033[91m"  # Red
    elif my_perf == target_perf:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[92m"  # Green

    print(f"{color}{method_name}: {my_perf}, target: {target_perf}\033[0m")


def store_comparison(dataset_name, run_id, store_path, method_name=DEFAULT_METHOD_NAME):
    prefix = f"{dataset_name}_"
    suffix = f"_{run_id}_sample1.json"
    prefix_len = len(prefix)
    suffix_len = len(suffix)

    def get_method_name(file_name):
        return file_name[prefix_len:-suffix_len]

    def get_performances(data):
        performances = {}

        # In the paper they show the max of the AggregationDetection-XGBClassifier among all tables
        interesting_stats = {
            "multi_table_metrics": [
                "AggregationDetection-XGBClassifier",
                # "ParentChildAggregationDetection-XGBClassifier",
                # "ParentChildDetection-XGBClassifier",
            ],
            "single_table_metrics": ["SingleTableDetection-XGBClassifier"],
        }

        if "multi_table_metrics" in data and "AggregationDetection-XGBClassifier" in data["multi_table_metrics"]:
            performances['Paper perf (max AD with XGBoost)'] = np.array(
                [p["accuracy"] for p in data["multi_table_metrics"]["AggregationDetection-XGBClassifier"].values()]
            ).max().round(3)

        for level1 in interesting_stats:
            for level2 in interesting_stats[level1]:
                if level1 in data and level2 in data[level1]:
                    perfs = data[level1][level2]
                    performances[level2 + "_max"] = np.array(
                        [p["accuracy"] for p in perfs.values()]
                    ).max().round(3)
                else:
                    performances[level2 + "_max"] = np.nan

        return performances

    folder = f"{RESULTS_PATH}/{run_id}"
    result_files = fnmatch.filter(
        sorted(os.listdir(folder)), pat=f"{dataset_name}_*.json"
    )
    data = {get_method_name(f): load_json(folder + "/" + f) for f in result_files}
    df = pd.DataFrame({method: get_performances(d) for method, d in data.items()})
    df = df.T

    stats = {
        stat: df[stat].sort_values(ascending=True).to_dict() for stat in df.columns
    }

    stats['target'] = TARGET_PERFORMANCES[dataset_name]

    store_json(stats, file=f"{store_path}/{dataset_name}_comparison.json")

    # Print the important numbers
    my_perf = df.loc[method_name]['Paper perf (max AD with XGBoost)']
    target_perf = TARGET_PERFORMANCES[dataset_name]

    # Use the new function to print performance
    print_performance(my_perf, target_perf, method_name)
    
    return {method_name: my_perf, "benchmark": target_perf}


def run(artifact_folder, method_name: str, full_test: bool = False, run_id: int = 1, generated_dataset_subfolder: str = "/report/generated"):
    dataset_name = get_dataset_name(artifact_folder)
    target_folder = "syntherela_benchmark/data/synthetic/" + dataset_name + "/" + method_name + f"/{run_id}/sample1"

    shutil.copytree(
        artifact_folder + generated_dataset_subfolder,
        target_folder,
        dirs_exist_ok=True,
    )
    
    restore_dates(target_folder)

    run_benchmark(
        dataset_name=dataset_name,
        methods=[method_name],
        run_id=run_id,
        full_test=full_test,
    )

    return store_comparison(
        dataset_name=dataset_name,
        run_id=run_id,
        store_path=artifact_folder,
        method_name=method_name,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Syntherela benchmark.")
    parser.add_argument(
        "--artifact_folder",
        type=str,
        help="Path to the artifact folder.",
        default=most_recent_artifact_folder(),
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=DEFAULT_METHOD_NAME,
        help="Chosen name of the method.",
    )
    parser.add_argument(
        "--full_test",
        action="store_true",
        help="Flag to indicate only relations should be processed.",
    )

    args = parser.parse_args()

    artifact_folder = args.artifact_folder
    method_name = args.method_name
    full_test = args.full_test
    
    run(artifact_folder, method_name, full_test)

