
from src.constants import MODELS_FOLDER
from src.experiment_config import load_experiment
from src.util import find, get_available_device

SEED = 1000

PATTERN = "_walmart*_x1_prediction"

last_experiment_path = find(str(MODELS_FOLDER), pattern=f"*{PATTERN}*")
print(f"\nLoading last experiment from {last_experiment_path}\n")

CONFIG = load_experiment(last_experiment_path)

CONFIG.train = False
CONFIG.name = 'loaded:' + CONFIG.name

    
CONFIG.generation_options = dict(n_samples=1, steps=100, x0_seed=None)
CONFIG.device = get_available_device()
