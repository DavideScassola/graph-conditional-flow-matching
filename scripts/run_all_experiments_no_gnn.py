# test_math_operations.py
import sys
import gc

sys.path.append(".")
from scripts.remove_duplicate_refs import remove_duplicates_from_artifact_folder
from src.experiment_controller import run_experiment

SEEDS = (1, 2, 3)
CONFIGS = (
    #"test/configs/sb_biodegradability_no_gnn.py",
    #"test/configs/sb_walmart_subsampled_no_gnn.py",
    "test/configs/sb_rossmann_subsampled_no_gnn.py",
    #"test/configs/sb_airbnb_no_gnn.py",
    "test/configs/sb_CORA_v1_no_gnn.py",
    "test/configs/sb_imdb_MovieLens_v1_no_gnn.py",
    )
    
folders = []

for config_file in CONFIGS:
    for seed in SEEDS:
        print(f"Running {config_file} with seed {seed}")
        folder = run_experiment(config_file, seed_override=seed, postfix=f"_paper_exp_no_gnn_seed_{seed}")
        folders.append(folder)
        if 'CORA' in folder:
            remove_duplicates_from_artifact_folder(folder, table_name="content.csv", columns=["paper_id", "word_cited_id"])
        gc.collect()
