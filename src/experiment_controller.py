import os
import shutil
import time
from pathlib import Path

from src.constants import CONFIG_FILE_NAME, MODELS_FOLDER
from src.util import create_experiment_folder, set_seeds, store_json

from .experiment_config import ExperimentConfig, getConfig, store_experiment


def create_models_folder() -> Path:
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    return MODELS_FOLDER


def store_results(
    *, config_file: str | Path, config: ExperimentConfig, folder_path: Path
):
    shutil.copyfile(config_file, folder_path / CONFIG_FILE_NAME)
    config.model.store(folder_path)
    config.model.generate_report(
        path=folder_path,
        dataset=config.dataset,
        generation_options=config.generation_options,
        device=config.device,
    )


def run_experiment(
    config_file: str, seed_override: int | None = None, postfix: str = ""
) -> str:
    start_time = time.time()

    config = getConfig(config_file)
    if config.seed:
        set_seeds(config.seed)
    if seed_override:
        set_seeds(seed_override)
    if config.train:
        config.model.train(config.dataset, device=config.device)

    folder_path = Path(
        create_experiment_folder(
            path=create_models_folder(), postfix=config.name + postfix
        )
    )

    store_results(config_file=config_file, config=config, folder_path=folder_path)
    if config.store:
        store_experiment(config, path=folder_path)

    end_time = time.time()
    time_lapsed = end_time - start_time

    store_json(
        file=folder_path / "time_lapsed.json",
        d={"time_lapsed_seconds": time_lapsed},
    )

    return str(folder_path)
