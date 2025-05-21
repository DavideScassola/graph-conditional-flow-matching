import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from syntherela.data import load_tables, remove_sdv_columns
from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection.aggregation_detection import (
    BaseAggregationDetection)
from tqdm import tqdm

from syntherela_benchmark.benchmark_artifact import DEFAULT_METHOD_NAME

# Code inspired from privacy_check.py from the SyntheRela Repository (https://github.com/martinjurkovic/syntherela)

QUANTILE_THRESHOLD = 0.02
REPETITIONS = 10

def select_smallest_n(x, n):
    smallest_indices = np.argsort(x)[:n]
    mask = np.zeros_like(x, dtype=bool)
    mask[smallest_indices] = True
    return x[mask]

def reduce_to_common_length(x, y):
    min_length = min(len(x), len(y))
    x = x[:min_length]
    y = y[:min_length]
    return x, y

def percentile_selection(dcrs_test, dcrs_syn, quantile):
    dcrs_test = dcrs_test[dcrs_test > 0]
    dcrs_syn = dcrs_syn[dcrs_syn > 0]
    q = np.quantile(dcrs_test, quantile)
    small_dcrs_test = dcrs_test[(dcrs_test <= q)]
    if False:
        small_dcrs_syn = select_smallest_n(dcrs_syn, len(small_dcrs_test))
    else:
        small_dcrs_syn = dcrs_syn[(dcrs_syn <= q)]
        small_dcrs_test, small_dcrs_syn = reduce_to_common_length(
            small_dcrs_test, small_dcrs_syn
        )
    #print('length of dcrs_test:', len(dcrs_test))

    return small_dcrs_test, small_dcrs_syn

def small_dcr_compare(dcrs_syn, dcrs_test, quantile):
    small_dcrs_test, small_dcrs_syn = percentile_selection(dcrs_test, dcrs_syn, quantile=quantile)
    diff = small_dcrs_syn - small_dcrs_test + np.random.normal(loc=0, scale=1e-10, size=small_dcrs_syn.shape
    )  # add small noise to avoid equality
    score = (diff>0).mean()
    return score

def dcr_privacy_score(dcrs_syn, dcrs_test, quantile):
    dcrs_syn_noise = dcrs_syn + np.random.uniform(low=0.0, high=1e-15, size=dcrs_syn.shape)
    dcrs_test_noise = dcrs_test + np.random.uniform(low=0.0, high=1e-15, size=dcrs_test.shape)
    q = np.quantile(dcrs_test_noise, quantile)
    dcr = (dcrs_syn_noise < q).sum() / (len(dcrs_test_noise)*quantile)
    return 100*quantile * (dcr-1)/(1-quantile)

def quantile_difference_score(dcrs_syn, dcrs_test, quantile):
    dcrs_syn_noise = dcrs_syn + np.random.uniform(low=0.0, high=1e-15, size=dcrs_syn.shape)
    dcrs_test_noise = dcrs_test + np.random.uniform(low=0.0, high=1e-15, size=dcrs_test.shape)
    q = np.quantile(dcrs_test_noise, quantile)
    return ((dcrs_syn_noise < q).mean() - quantile)*100

def synth_below_quantile_percentage(dcrs_syn, dcrs_test, quantile):
    dcrs_syn_noise = dcrs_syn + np.random.uniform(low=0.0, high=1e-15, size=dcrs_syn.shape)
    dcrs_test_noise = dcrs_test + np.random.uniform(low=0.0, high=1e-15, size=dcrs_test.shape)
    q = np.quantile(dcrs_test_noise, quantile)
    return (dcrs_syn_noise < q).mean()*100
    

def load_data(dataset_name, run=1, data_path="syntherela_benchmark/data"):
    metadata = Metadata().load_from_json(
        f"{data_path}/original/{dataset_name}/metadata.json"
    )

    tables = load_tables(f"{data_path}/original/{dataset_name}/", metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic = load_tables(
        f"{data_path}/synthetic/{dataset_name}/{DEFAULT_METHOD_NAME}/{run}/sample1", metadata
    )

    return tables, tables_synthetic, metadata


def prepare_data(table_real, table_syn, metadata, table_name):
    id_columns = metadata.get_column_names(table_name, sdtype="id")
    numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")
    categorical_columns = metadata.get_column_names(table_name, sdtype="categorical")
    datetime_columns = metadata.get_column_names(table_name, sdtype="datetime")

    for column in categorical_columns:
        table_real[column] = pd.Categorical(table_real[column])
        table_syn[column] = pd.Categorical(
            table_syn[column], categories=table_real[column].cat.categories
        )

    table_real.drop(columns=id_columns, inplace=True)
    table_syn.drop(columns=id_columns, inplace=True)

    for column in datetime_columns:
        table_real[column] = (
            pd.to_datetime(table_real[column]).astype(np.int64) // 10**9
        )
        table_syn[column] = pd.to_datetime(table_syn[column]).astype(np.int64) // 10**9

    numerical_columns += datetime_columns

    table_real[numerical_columns] = table_real[numerical_columns].fillna(0)
    table_syn[numerical_columns] = table_syn[numerical_columns].fillna(0)

    table_real = pd.get_dummies(table_real, columns=categorical_columns).astype(
        np.float32
    )
    table_syn = pd.get_dummies(table_syn, columns=categorical_columns).astype(
        np.float32
    )

    table_train, table_test = train_test_split(
        table_real, test_size=0.5, random_state=42
    )

    if len(numerical_columns) > 0:
        scaler = StandardScaler()
        scaler.fit(table_train[numerical_columns])

        table_train[numerical_columns] = scaler.transform(table_train[numerical_columns])
        table_test[numerical_columns] = scaler.transform(table_test[numerical_columns])
        table_syn[numerical_columns] = scaler.transform(table_syn[numerical_columns])

    return table_train, table_test, table_syn


def evaluate_dcr(table_train, table_syn, table_test, seed = None):
    if seed is not None:
        np.random.seed(seed)
    real_data = table_train.values
    syn_data = table_syn.sample(n=table_test.shape[0], replace=True).values
    test_data = table_test.values
    

    dcrs_syn = []
    dcrs_test = []
    batch_size = 4000

    for i in range((syn_data.shape[0] // batch_size) + 1):
        syn_data_batch = syn_data[i * batch_size : (i + 1) * batch_size]
        test_data_batch = test_data[i * batch_size : (i + 1) * batch_size]
        if syn_data_batch.shape[0] == 0:
            break

        dcr_syn = pairwise_distances(syn_data_batch, real_data, metric="euclidean").min(
            axis=1
        )
        dcr_test = pairwise_distances(
            test_data_batch, real_data, metric="euclidean"
        ).min(axis=1)

        dcrs_syn.append(dcr_syn)
        dcrs_test.append(dcr_test)

    dcrs_syn = np.concatenate(dcrs_syn)
    dcrs_test = np.concatenate(dcrs_test)
    
    zero_percentage = (dcrs_test == 0.0).mean()
    #print(f"fraction of identical: {zero_percentage:.2f}")

    #score = small_dcr_compare(dcrs_syn, dcrs_test, quantile=0.02)

    #score = (dcrs_test <= dcrs_syn).mean()
    return dcrs_syn, dcrs_test #, score


def estimate_dcr_score(tables, tables_synthetic, table_name, metadata, m=10, seed=42):
    table_real = tables[table_name]
    table_syn = tables_synthetic[table_name]
    table_syn = table_syn[table_real.columns]
    table_train, table_test, table_syn = prepare_data(
        table_real.copy(), table_syn.copy(), metadata, table_name
    )
    #scores = []
    dcr_syn = []
    dcr_test = []
    
    # show dimensions of the tables
    #print(f"Real table shape: {table_real.shape}")
    #print(f"Synthetic table shape: {table_syn.shape}")
    #print(f"Train table shape: {table_train.shape}")
    #print(f"Test table shape: {table_test.shape}")

    
    for i in tqdm(range(m)):
        dcrs_syn, dcrs_test = evaluate_dcr(
            table_train, table_syn, table_test, seed=seed + i
        )
        #scores.append(score)
        dcr_syn.append(dcrs_syn)
        dcr_test.append(dcrs_test)
        
    #result = f"Mean score: {np.mean(scores):.4f} +- {np.std(scores) / np.sqrt(len(scores)) :.4f}"

    #print(
    #    result
    #)
    dcr_syn = np.stack(dcr_syn)
    dcr_test = np.stack(dcr_test)
    return dcr_syn, dcr_test#, result

def main(dataset_name, run: int, aggregate: bool, metrics):
    
    tables, tables_synthetic, metadata = load_data(dataset_name, run)
    
    if aggregate:
        tables, updated_metadata = BaseAggregationDetection.add_aggregations(
            tables, deepcopy(metadata)
        )
        tables_synthetic, _ = BaseAggregationDetection.add_aggregations(
            tables_synthetic, metadata, update_metadata=False
        )
        metadata = updated_metadata
    
    scores_dict = {}
    shape_dict = {}

    for table_name in tables.keys():
        
        print(f"Evaluating table: {table_name}")
        
        # Check if the only non-id column is categorical
        skip_table = (len(metadata.get_column_names(table_name, sdtype="categorical")) + len(metadata.get_column_names(table_name, sdtype="numerical"))) <= 1
            
        if skip_table:
            print(f"not enough columns in {table_name}. Skipping...")
        else:
            os.makedirs("syntherela_benchmark/dcrs", exist_ok=True)
            cached_dcrs_path = f"syntherela_benchmark/dcrs/dcrs_{dataset_name}_{table_name}_run{run}_{REPETITIONS}reps.npz"
            if aggregate:
                cached_dcrs_path = f"syntherela_benchmark/dcrs/aggregated_dcrs_{dataset_name}_{table_name}_run{run}_{REPETITIONS}reps.npz"
            if os.path.exists(cached_dcrs_path):
                dcrs = np.load(cached_dcrs_path)
            else:
                dcrs_syn, dcrs_test = estimate_dcr_score(
                    tables, tables_synthetic, table_name, metadata, m=REPETITIONS, seed=1
                )
                np.savez(cached_dcrs_path, dcrs_syn=dcrs_syn, dcrs_test=dcrs_test)
                dcrs = np.load(cached_dcrs_path)
                
            #dcr_test, dcr_syn = percentile_selection(dcr_test, dcr_syn, quantile=0.05)
            dcr_syn = dcrs["dcrs_syn"].flatten()
            dcr_test = dcrs["dcrs_test"].flatten()
            
            
            if (dcr_test==0).mean() > 0.1:
                print(f"Too many identical rows in {table_name}: Skipping...")
                continue
            
            scores = {}
            for metric in metrics:
                scores[metric] = [eval(metric)(dcrs["dcrs_syn"][i], dcrs["dcrs_test"][i], quantile=QUANTILE_THRESHOLD) for i in range(len(dcrs["dcrs_syn"]))]
             

                        

            nbins = 100
            max_val = np.quantile(dcr_syn, 0.975)
            max_val = np.quantile(dcr_test[dcr_test>0], QUANTILE_THRESHOLD)
            
            if dataset_name == 'rossmann_subsampled':
                pass

            plt.figure(figsize=(10, 10))
            bins = plt.hist(dcr_test, bins=nbins, range=(0, max_val), alpha=0.5, label="Real")
            plt.hist(dcr_syn, bins=bins[1], range=(0, max_val), alpha=0.5, label="Synthetic")
            plt.legend(fontsize=15)
            plt.savefig(f"syntherela_benchmark/results/{'aggregated' if aggregate else ''}_dcr_{dataset_name}_{table_name}_run{run}.png")
            plt.close()
            
            scores_dict[table_name] = scores
            shape_dict[table_name] = tables[table_name].shape

    return scores_dict, shape_dict


if __name__ == "__main__":
    datasets = (
        "airbnb-simplified_subsampled",
        "rossmann_subsampled",
        "walmart_subsampled",
        "imdb_MovieLens_v1",
        "Biodegradability_v1",
        "CORA_v1",
    )

    results = []

    AGGREGATE = True
    METRICS = ["synth_below_quantile_percentage", "dcr_privacy_score"]

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for run in (1, 2, 3):
            scores_dict, shape_dict = main(dataset_name=dataset, run=run, metrics=METRICS, aggregate=AGGREGATE)
            for table_name, scores in scores_dict.items():
                row = {
                    "dataset": dataset,
                    "table": table_name,
                    "run": run,
                    "n_rows": shape_dict[table_name][0],
                    "n_cols": shape_dict[table_name][1],
                }
                for metric, score in scores.items():
                    row[f"mean_{metric}"] = np.mean(np.array(score))
                    row[f"std_{metric}"] = np.std(np.array(score))

                results.append(row)

    # Create a DataFrame from the results
    df = pd.DataFrame(results).sort_values(by=["dataset", "table", "run"])
    df.to_csv(f"privacy_results_quantile_{str(QUANTILE_THRESHOLD)}{'_aggregated' if AGGREGATE else ''}.csv", index=False)

    # Group by dataset and table, and compute mean and std for each metric
    agg_dict = {}
    for metric in METRICS:
        agg_dict[f"mean_{metric}"] = ("mean_" + metric, "mean")
        agg_dict[f"std_{metric}"] = ("mean_" + metric, "std")

    df_agg = df.groupby(["dataset", "table", "n_rows", "n_cols"]).agg(**agg_dict).reset_index()

    # Save the aggregated results
    print("\nAggregated Results:")
    print(df_agg.round(2))
    df_agg.to_csv(f"agg_privacy_results_quantile_{str(QUANTILE_THRESHOLD)}{'_aggregated' if AGGREGATE else ''}.csv", index=False)

