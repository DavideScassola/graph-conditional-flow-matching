import _thread
import fnmatch
import importlib.util
import io
import itertools
import json
import os
import pickle
import random
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict

import GPUtil
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor
from torch_geometric.data import HeteroData

INT_IS_NUMERICAL_THRESHOLD = 20


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


class edit_json:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if os.path.exists(self.filename):
            self.file = open(self.filename, "r+")
            self.data = json.load(self.file)
            # Move the pointer to the beginning of the file
            self.file.seek(0)
        else:
            self.file = open(self.filename, "w")
            self.data = {}
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        json.dump(self.data, self.file, indent=4)
        self.file.close()


def file_name(file: str | Path) -> str:
    return str(file).split("/")[-1]


def load_module(path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("module", path)
    if spec == None:
        raise ValueError(f"{path} is not a valid module path")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def execute_python_script(path: str | Path):
    os.system(f"python {path}")


def create_experiment_folder(*, path: Path, postfix: str | None = None) -> Path:
    postfix = f"_{postfix}" if postfix else ""
    folder_name = Path(datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + postfix)
    experiment_folder = path / folder_name
    os.makedirs(experiment_folder)
    return experiment_folder


@contextmanager
def requires_grad(X, *, reset_grad: bool):
    original_state = X.requires_grad
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    X.requires_grad = True
    if reset_grad:
        X.grad = None
    try:
        yield X
    finally:
        X.requires_grad = original_state
        torch.set_grad_enabled(was_grad_enabled)
        if reset_grad:
            X.grad = None


def gradient(
    *, f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor
) -> torch.Tensor:
    with requires_grad(X, reset_grad=True):
        value = f(X).sum()
        value.backward()
        # TODO: check if gradient flows only with respect to X!
        assert torch.isfinite(value), f(X)
        grad = X.grad
    return grad  # type: ignore


def find(folder: str, *, pattern: str = "*", index: int = -1) -> str:
    """
    Picks the last stored model
    """
    matches = fnmatch.filter(sorted(os.listdir(folder)), pattern)
    if matches:
        return f"{folder}/{matches[index]}"
    else:
        raise ValueError(f"No model found matching {pattern}")


def D_kl(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = min(len(x), len(y)) // 20

    def density_estimator(k):
        b = np.histogram(k, density=False, range=bounds, bins=bins)[0] + 1
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)
    return np.sum(np.where(px != 0, px * (np.log(px) - np.log(py)), 0))


def max_symmetric_D_kl(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = min(len(x), len(y)) // 20

    def density_estimator(k):
        b = (
            np.histogram(k, density=False, range=bounds, bins=bins)[0] + 1
        )  # +1 in order to have numerical stability
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)

    dkl = lambda p1, p2: np.sum(np.where(p1 != 0, p1 * (np.log(p1) - np.log(p2)), 0))
    return max(dkl(px, py), dkl(py, px))


def l1_divergence(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = int(np.sqrt(min(len(x), len(y))))

    def density_estimator(k):
        b = np.histogram(k, density=False, range=bounds, bins=bins)[0]
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)

    return np.sum(np.abs(px - py)) / 2


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def input_listener():
    def input_thread(a_list):
        try:
            input()
            a_list.append(True)
        except EOFError:
            print("EOFError: No input available.")
            a_list.append(False)

    a_list = []
    _thread.start_new_thread(input_thread, (a_list,))
    return a_list


def normalize(x: Tensor, dim):
    return (x - x.mean(dim=dim)) / x.std(dim=dim)


def pickle_load(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def pickle_store(object, *, file: str):
    with open(file, "wb") as f:
        return pickle.dump(object, f)


def get_available_device(mem_required: float = 0.05, verbose: bool = False):
    if not torch.cuda.is_available():
        return "cpu"

    try:
        devices = GPUtil.getGPUs()

        device_usages = [
            (device.id, device.memoryUsed / device.memoryTotal) for device in devices
        ]

        device_usages.sort(key=lambda x: x[1])

        if device_usages[0][1] > 1 - mem_required:
            return "cpu"

        out = "cuda:" + str(device_usages[0][0])
        if verbose:
            print("\033[92m" + f"Using {out}" + "\033[0m")
        return out
    except ValueError as e:
        print(e)
        return "cpu"


def cross_entropy(*, logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross entropy loss where the logits dimension is the last one, everything before is batched (differently from base working of torch cross entropy)
    """

    return torch.nn.functional.cross_entropy(
        logits.permute(0, -1, 1) if len(logits.shape) > 2 else logits,
        targets,
        reduction="none",
    )


def categorical_l1_histogram_distance(x, y) -> float:
    px = pd.Series(x).value_counts(normalize=True)
    py = pd.Series(y).value_counts(normalize=True)
    return abs(px.subtract(py, fill_value=0)).sum() / 2


def normalized_mutual_information_matrix(df: pd.DataFrame):
    # TODO: could be vectorized
    n = len(df.columns)
    nmi_matrix = np.ones((n, n))
    df_temp = df.astype(str)
    df_temp.fillna("[MISSING]", inplace=True)
    for i in range(n):
        for j in range(i):
            nmi_matrix[i, j] = normalized_mutual_info_score(
                df_temp.iloc[:, i], df_temp.iloc[:, j]
            )
            nmi_matrix[j, i] = nmi_matrix[i, j]
    return pd.DataFrame(nmi_matrix, columns=df.columns, index=df.columns)


def is_int(c: pd.Series) -> bool:
    """
    This function is used to determine if an int column is really numerical or used as categorical.
    Obviously it's not perfect, but it's a good heuristic.
    """
    if c.dtype != int:
        return False
    unique_values = c.nunique()
    range_values = c.max() - c.min() + 1
    # TODO: this is probably not possible and wrong, could cause issues
    return (
        unique_values < range_values or range_values > INT_IS_NUMERICAL_THRESHOLD
    ) and (unique_values > INT_IS_NUMERICAL_THRESHOLD)


def is_numerical(c: pd.Series) -> bool:
    return (c.dtype == float) or is_int(c)


def is_categorical(s: pd.Series) -> bool:
    if s.dtype == "object":
        return True
    return not is_numerical(s)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: not is_numerical(df[c]), df.columns))


def numerical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: is_numerical(df[c]), df.columns))


def only_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[categorical_columns(df)]


def only_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[numerical_columns(df)]


def find_primary_key(df: pd.DataFrame) -> str | None:
    # TODO: This doesn't work well
    for c in df.columns:
        if df[c].nunique() == len(df) and ("id" == c[-2:].lower()):
            return c
    return None


def find_foreign_keys(
    df: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, str]:
    foreign_keys = {}
    primary_key = find_primary_key(df)
    for name, other_df in other_dfs.items():
        other_primary_key = find_primary_key(other_df)
        if other_primary_key is None:
            continue
        for c in df.columns:
            if c != primary_key and c == other_primary_key:
                foreign_keys[c] = {"table": name, "key_column": c}
    return foreign_keys


def get_indices_from_column(
    df: pd.DataFrame, *, alternative_index: list, column: str
) -> list:
    """
    Get the indices of the specified values in the given column of the DataFrame.

    :param df: The DataFrame.
    :param values: The list of values to find indices for.
    :param column: The column to search for the values.
    :return: A list of indices corresponding to the values.
    """
    return pd.Series(df.index, index=df[column])[alternative_index].tolist()


def get_hetero_data_device(hetero_data: HeteroData):
    return list(hetero_data[hetero_data.metadata()[0][0]].values())[0].device


def infer_int_series_type(series: pd.Series) -> str:
    unique_values = series.nunique()
    range_values = series.max() - series.min() + 1
    if unique_values < range_values or unique_values > INT_IS_NUMERICAL_THRESHOLD:
        return "numerical_int"
    return "categorical_int"


def infer_series_type(series: pd.Series) -> str:

    # First the easy ones
    if series.dtype == "float":
        return "float"

    if series.dtype in ("object", "category"):
        return "string"

    if series.dtype == "bool":
        return "bool"

    # Now the ambiguous ones
    if series.dtype == "int":
        return infer_int_series_type(series)

    raise ValueError(f"Could not infer type of series {series.name} unique values")


def log_marginals(*, one_hot_data: Tensor, logit_minimum: float = -1000):
    prob = one_hot_data.float().mean(0)
    return torch.where(
        prob > 0,
        torch.log(prob),
        logit_minimum,
    ).to(one_hot_data.device)


@contextmanager
def animated_message(message):
    spinner = itertools.cycle(["-", "\\", "|", "/"])
    stop_event = threading.Event()

    def animate():
        with tqdm.tqdm(bar_format="{desc}", ncols=100, leave=False) as pbar:
            while not stop_event.is_set():
                pbar.set_description(f"{message} {next(spinner)}")
                pbar.update(0.1)
                time.sleep(0.1)
            pbar.set_description(f"{message} done")
            pbar.update(1)
            pbar.close()

    thread = threading.Thread(target=animate)
    thread.start()

    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        print("\r", end="")  # Clear the line after the animation


def fake_float_to_int(df: pd.DataFrame) -> None:
    for column in df.columns:
        if df[column].dtype == float and (~df[column].isna()).all():
            s_int = df[column].astype(int)
            if (df[column] == s_int).all():
                df[column] = s_int


def exponential_decaying_lr(epoch, start_lr, final_lr, n_epochs):
    return (final_lr / start_lr) ** (epoch / n_epochs)


def save_model_print(model, file_path):
    buf = io.StringIO()
    print(model, file=buf)
    with open(file_path, "w") as f:
        f.write(buf.getvalue())
