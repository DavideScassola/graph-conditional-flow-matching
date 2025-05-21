import os
import re
import fnmatch
import json
import csv
import argparse
import pandas as pd

def extract_date_and_expname(folder_name):
    # Example: 2025-05-16_21:56:40_455310_imdb_MovieLens_v1_paper_exp_seed_3
    m = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}_\d+)_([^/]+)", folder_name)
    if m:
        date = m.group(1)
        exp_name = m.group(2)
        return date, exp_name
    return "", folder_name

def extract_dataset_name(config_path):
    try:
        with open(config_path, "r") as f:
            for line in f:
                m = re.search(r'NAME\s*=\s*["\']([^"\']+)["\']', line)
                if m:
                    return m.group(1)
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
    return ""

def extract_time_lapsed(time_path):
    try:
        with open(time_path, "r") as f:
            data = json.load(f)
            return data.get("time_lapsed_seconds", "")
    except Exception as e:
        print(f"Error reading {time_path}: {e}")
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="fnmatch pattern for experiment folders")
    parser.add_argument("--root_dir", default="artifacts/models", help="Root directory to search")
    parser.add_argument("--output_csv", default="experiment_summary.csv", help="Output CSV file")
    args = parser.parse_args()

    results = []

    for folder in os.listdir(args.root_dir):
        if not fnmatch.fnmatch(folder, args.pattern):
            continue
        exp_dir = os.path.join(args.root_dir, folder)
        if not os.path.isdir(exp_dir):
            continue

        date, exp_name = extract_date_and_expname(folder)
        config_path = os.path.join(exp_dir, "config.py")
        dataset_name = extract_dataset_name(config_path)
        time_path = os.path.join(exp_dir, "time_lapsed.json")
        time_lapsed = extract_time_lapsed(time_path)

        results.append((date, exp_name, dataset_name, time_lapsed))

    # Sort by date
    results.sort(key=lambda x: x[0])

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "experiment_name", "dataset_name", "time_lapsed_seconds"])
        writer.writerows(results)

    print(f"CSV written to {args.output_csv}")
    df = pd.read_csv(args.output_csv)
    df_max = df.groupby("dataset_name").agg({"time_lapsed_seconds": "max"}).reset_index().sort_values(by="dataset_name", ascending=True)
    print(df_max)

if __name__ == "__main__":
    main()