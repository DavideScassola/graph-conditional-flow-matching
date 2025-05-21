# test_math_operations.py
import fnmatch
import json
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(".")

def find_files_with_pattern(directory, pattern):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                matching_files.append(os.path.join(root, file))
    return matching_files

def json_to_record(file: str)-> dict:
    with open(file, 'r') as f:
        data = json.load(f)
    dataset = os.path.basename(file).split('_')[0]
    method_name = file.split('_')[-3]
    run_id = int(file.split('_')[-2])
    
    if "multi_table_metrics" in data and "AggregationDetection-XGBClassifier" in data["multi_table_metrics"]:
        relevant_stat = np.array(
            [p["accuracy"] for p in data["multi_table_metrics"]["AggregationDetection-XGBClassifier"].values()]
        ).max().round(4)
    else:
        return {}
    
    #relevant_stat = np.array(
    #            [p["accuracy"] for p in data["multi_table_metrics"]["AggregationDetection-XGBClassifier"].values()]
    #        ).max().round(3) if "multi_table_metrics" in data and "AggregationDetection-XGBClassifier" in data["multi_table_metrics"] else 0.0
    return {
        'method_name': method_name,
        'dataset': dataset,
        'run_id': run_id,
        'AD_max': relevant_stat}

SR_RESULTS_PATH = 'syntherela_benchmark/results'

result_files = find_files_with_pattern(SR_RESULTS_PATH, '*.json')
records = [json_to_record(file) for file in result_files]
df = pd.DataFrame(records)
df = df.sort_values(by=['dataset', 'method_name', 'run_id'])

pt = df.pivot_table(
    index=['dataset', 'method_name'],
    values=['AD_max'],
    aggfunc=['mean', 'std', 'min', 'median', 'max'],
).round(4)

print(pt)

pt.to_csv('pivot_table.csv')