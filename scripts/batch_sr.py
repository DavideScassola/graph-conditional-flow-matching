# test_math_operations.py
from ast import pattern
import fnmatch
import sys
import os

sys.path.append(".")
from syntherela_benchmark.benchmark_artifact import run as get_benchmark_stat
from syntherela_benchmark.benchmark_artifact import DEFAULT_METHOD_NAME

DEFAULT_PATTERN = '*_paper_exp*'
pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATTERN

    
folders = fnmatch.filter(sorted(os.listdir('artifacts/models')), pattern)

for folder in folders:
    folder = os.path.join('artifacts/models', folder)
    if not fnmatch.filter(sorted(os.listdir(folder)), '*comparison*.json'):
        postfix = 'NoGNN' if 'no_gnn' in folder else ''
        run_id = int(folder.split("_")[-1])
        print(f"Evaluating {folder}")
        results = get_benchmark_stat(folder, method_name=DEFAULT_METHOD_NAME+postfix, run_id=run_id)
        if "CORA" in folder:
            results = get_benchmark_stat(folder, method_name=DEFAULT_METHOD_NAME+'Original'+postfix, run_id=run_id, generated_dataset_subfolder='/report/generated_old')