# test_math_operations.py
import sys
import gc

sys.path.append(".")
from scripts.remove_duplicate_refs import remove_duplicates_from_artifact_folder
from src.experiment_controller import run_experiment

SEEDS = (1, 2,3)
CONFIGS = (
    "test/configs/sb_biodegradability.py",
    "test/configs/sb_walmart_subsampled.py",
    "test/configs/sb_rossmann_subsampled.py",
    "test/configs/sb_airbnb.py",
    "test/configs/sb_CORA_v1.py",
    "test/configs/sb_imdb_MovieLens_v1.py",
    )
    
folders = []

for config_file in CONFIGS:
    for seed in SEEDS:
        print(f"Running {config_file} with seed {seed}")
        folder = run_experiment(config_file, seed_override=seed, postfix=f"_paper_exp_seed_{seed}")
        folders.append(folder)
        if 'CORA' in folder:
            remove_duplicates_from_artifact_folder(folder, table_name="content.csv", columns=["paper_id", "word_cited_id"])
        gc.collect()
