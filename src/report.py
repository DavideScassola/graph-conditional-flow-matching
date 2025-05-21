from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from matplotlib import image
from scipy.stats import kstest

from src.util import (animated_message, categorical_l1_histogram_distance,
                      edit_json, l1_divergence, load_json, max_symmetric_D_kl,
                      normalized_mutual_information_matrix,
                      only_categorical_columns, only_numerical_columns,
                      store_json)
from src.xgboost_discriminator import xgboost_discriminator_metrics_with_kfold

SAMPLES_NAME = "samples.csv"
CORRELATIONS_NAME = "correlations"
HISTOGRAMS_NAME = "histograms"
COORDINATES_NAME = "coordinates"
STATS_NAME = "stats.json"
SUMMARY_STATS_NAME = "stats_summary.json"
REPORT_FOLDER_NAME = "report"
IMAGE_FORMAT = "png"
NMI_PLOT_NAME = "nmi"


def kstest_pvalue_and_bool(x, y):
    result = kstest(x, y)
    return (1 if result.pvalue > 0.1 else 0, result.pvalue)


def list_corr(df):
    m = df.corr().to_numpy()
    return m[np.triu_indices_from(m, 1)]


def correlation_matrix_similarities(df1: pd.DataFrame, df2: pd.DataFrame):
    c1 = list_corr(df1)
    c2 = list_corr(df2)

    return {
        "average l1": np.mean(abs(c1 - c2)),
        "median l1": np.median(abs(c1 - c2)),
    }


def normalized_mutual_information_matrix_similarities(
    df1: pd.DataFrame, df2: pd.DataFrame
):
    c1 = normalized_mutual_information_matrix(df1)
    c2 = normalized_mutual_information_matrix(df2)

    return {
        "average l1": np.mean(abs(c1 - c2)),
        "median l1": np.median(abs(c1 - c2)),
    }


NUMERICAL_STATS = {
    "mean": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0, ddof=1),
    "correlations": lambda df: df.corr(),
}

COLUMNWISE_NUMERICAL_COMPARISON_STATS = {
    "l1_divergence": l1_divergence,
    "max_symmetric_D_kl": max_symmetric_D_kl,
    "two-sample Kolmogorov-Smirnov test": kstest_pvalue_and_bool,
}

GLOBAL_NUMERICAL_COMPARISON_STATS = {
    "correlation matrix similarity": correlation_matrix_similarities
}

COLUMNWISE_CATEGORICAL_COMPARISON_STATS = {
    "l1_histogram_distance": categorical_l1_histogram_distance,
}

GLOBAL_CATEGORICAL_COMPARISON_STATS = {
    "NMI matrix similarity": normalized_mutual_information_matrix_similarities
}

CATEGORICAL_STATS = {}


def kde_plot(
    samples: pd.Series,
    *,
    ax=None,
    kde_args: dict = {},
    plt_args: dict = {},
    start=None,
    end=None,
):
    ax = ax if ax else plt.gca()
    sns.kdeplot(samples, ax=ax, **kde_args, **plt_args)
    if start is not None or end is not None:
        ax.set_xlim(start, end)


def coordinates_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    color = {
        name_original: "blue",
        name_generated: "orange",
    }

    df = {
        name_original: df_train.sample(frac=1),
        name_generated: df_generated,
    }

    n = min(len(df_generated), len(df_train))

    for name in (name_original, name_generated):
        plt.scatter(
            df[name].iloc[:n, 0],
            df[name].iloc[:n, 1],
            alpha=0.5,
            color=color[name],
            label=name,
            s=0.1,
        )

    plt.legend()
    plt.savefig(path / Path(f"{COORDINATES_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def bins(s: pd.Series):
    return np.linspace(min(s), max(s), int(len(s) ** 0.5))


def histograms_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    n = len(df_generated.columns)
    fig, axes = plt.subplots(n, 1, sharex=False)
    if n == 1:
        axes = [axes]
    fig.set_size_inches(6, n + 4)
    fig.suptitle("Histogram Comparison")

    plt.tight_layout()

    for i, c in enumerate(df_generated.columns):
        is_categorical = df_train[c].dtype in (str, "O", bool, "category")

        if is_categorical:
            s_train = df_train[c]
            s_generated = df_generated[c]

            common_args = dict(
                x=c,
                stat="proportion",
                alpha=0.5,
            )
        else:
            s_train = df_train[c].dropna()  # TODO: some information is lost
            s_generated = df_generated[c].dropna()  # TODO: some information is lost
            data = pd.Series(pd.concat((s_generated, s_train)))
            std = data.std()
            start = np.min(data.to_numpy()) - 0.5 * std
            end = np.max(data.to_numpy()) + 0.5 * std
            b = bins(pd.Series(pd.concat((s_generated, s_train))))
            start = start
            end = end
            common_args = dict(x=c, stat="density", alpha=0.5, bins=b)
            if s_train.std() == 0.0:
                del common_args["bins"]

        ax_true = sns.histplot(
            ax=axes[i], data=df_train, label=name_original, **common_args
        )

        if is_categorical:
            ax_generated = sns.histplot(
                ax=axes[i], data=df_generated, label=name_generated, **common_args
            )
        else:
            bandwidth = min(s_train.std(), s_generated.std()) / 10

            if s_train.std() > 0:
                kde_plot(
                    s_train,
                    ax=axes[i],
                    start=start,  # type: ignore
                    end=end,  # type: ignore
                    kde_args={"bw_adjust": bandwidth},
                    plt_args={"color": "blue", "alpha": 0.5, "linestyle": "-"},
                )

            ax_generated = sns.histplot(
                ax=axes[i], data=df_generated, label=name_generated, **common_args
            )
            if s_generated.std() > 0:
                kde_plot(
                    s_generated,
                    ax=axes[i],
                    start=start,  # type: ignore
                    end=end,  # type: ignore
                    kde_args={"bw_adjust": bandwidth},
                    plt_args={"color": "orange", "alpha": 0.8, "linestyle": "-"},
                )

    plt.legend()
    plt.savefig(path / Path(f"{HISTOGRAMS_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def store_samples(*, df_generated: pd.DataFrame, path: Path, name: str = SAMPLES_NAME):
    df_generated.fillna("").to_csv(path / Path(name), index=False, na_rep="")


def get_stats(df: pd.DataFrame):
    df_numerical = only_numerical_columns(df)
    df_categorical = only_categorical_columns(df)

    assert len(df_numerical.columns) + len(df_categorical.columns) == len(
        df.columns
    ), "Some columns are not considered numerical or categorical"

    numerical_stats = (
        {k: stat(df_numerical).to_dict() for k, stat in NUMERICAL_STATS.items()}
        if len(df_numerical.columns) > 0
        else {}
    )

    categorical_stats = (
        {k: stat(df_categorical).to_dict() for k, stat in CATEGORICAL_STATS.items()}
        if len(df_categorical.columns) > 0
        else {}
    )

    return {**numerical_stats, **categorical_stats}


def get_columnwise_comparison_stats(df1: pd.DataFrame, df2: pd.DataFrame):
    df1_numerical = only_numerical_columns(df1)
    df2_numerical = df2[df1_numerical.columns]

    out = {}

    if len(df1_numerical.columns):
        out["numerical_stats"] = {
            k: {
                column: stat(df1_numerical[column], df2_numerical[column])
                for column in df1_numerical.columns
            }
            for k, stat in COLUMNWISE_NUMERICAL_COMPARISON_STATS.items()
        }

    df1_categorical = only_categorical_columns(df1)
    df2_categorical = df2[df1_categorical.columns]

    if len(df1_categorical.columns):
        out["categorical_stats"] = (
            {
                k: {
                    column: stat(df1_categorical[column], df2_categorical[column])
                    for column in df1_categorical.columns
                }
                for k, stat in COLUMNWISE_CATEGORICAL_COMPARISON_STATS.items()
            }
            # if len(df1_categorical.columns) > 1
            # else {}
        )

    return out


def get_global_comparison_stats(df_train: pd.DataFrame, df_generated: pd.DataFrame):
    df_numerical = only_numerical_columns(df_train)
    df_numerical_generated = df_generated[df_numerical.columns]

    numerical_stats = (
        {
            k: stat(df_numerical, df_numerical_generated)
            for k, stat in GLOBAL_NUMERICAL_COMPARISON_STATS.items()
        }
        if len(df_numerical.columns) > 1
        else {}
    )

    df_categorical = only_categorical_columns(df_train)
    df_categorical_generated = only_categorical_columns(df_generated)

    categorical_stats = (
        {
            k: stat(df_categorical, df_categorical_generated)
            for k, stat in GLOBAL_CATEGORICAL_COMPARISON_STATS.items()
        }
        if len(df_categorical.columns) > 1
        else {}
    )

    return {"numerical_stats": numerical_stats, "categorical_stats": categorical_stats}


def statistics_comparison(
    *,
    df: dict,
    file: Path,
    **kwargs,
):
    df_generated, df_train = list(df.values())
    name_generated, name_original = list(df.keys())

    generated_stats = get_stats(df_generated)
    true_data_stats = get_stats(df_train)
    columnwise_comparison_stats = get_columnwise_comparison_stats(
        df1=df_train, df2=df_generated
    )
    global_comparison_stats = (
        get_global_comparison_stats(df_train=df_train, df_generated=df_generated)
        if len(df_train.columns) > 1
        else None
    )
    stats = {
        name_generated: generated_stats,
        name_original: true_data_stats,
        "columnwise_comparison": columnwise_comparison_stats,
    }

    if global_comparison_stats:
        stats["global_comparison"] = global_comparison_stats
    store_json(stats, file=file)


def correlations_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    df_generated_numerical = df_generated.select_dtypes(include="number")
    df_train_numerical = df_train.select_dtypes(include="number")

    n = len(df_generated_numerical.columns)
    plt.figure().set_size_inches(n / 2 + 4, n / 2)

    if n < 2:
        return
    bi_corr = np.tril(df_generated_numerical.corr()) + np.triu(
        df_train_numerical.corr(), k=1
    )
    df = pd.DataFrame(
        bi_corr,
        index=df_generated_numerical.columns,
        columns=df_generated_numerical.columns,
    )
    sns.heatmap(
        df,
        annot=True,
        vmin=-1.0,
        vmax=1.0,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        alpha=0.8,
    )

    plt.title(f"Correlations (lower: {name_generated}, upper: {name_original})")
    plt.savefig(path / Path(f"{CORRELATIONS_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def mutual_information_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    df_generated_categorical = df_generated.select_dtypes(exclude="number")
    df_train_categorical = df_train.select_dtypes(exclude="number")

    n = len(df_generated_categorical.columns)
    plt.figure().set_size_inches(n / 2 + 4, n / 2)

    if n < 2:
        return
    generated_nmi = normalized_mutual_information_matrix(df_generated_categorical)
    original_nmi = normalized_mutual_information_matrix(df_train_categorical)

    bi_corr = np.tril(generated_nmi) + np.triu(original_nmi, k=1)
    df = pd.DataFrame(
        bi_corr,
        index=df_generated_categorical.columns,
        columns=df_generated_categorical.columns,
    )
    sns.heatmap(
        df,
        annot=True,
        vmin=-1.0,
        vmax=1.0,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        alpha=0.8,
    )

    plt.title(f"NMI (lower: {name_generated}, upper: {name_original})")
    plt.savefig(path / Path(f"{NMI_PLOT_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def store_images(samples: torch.Tensor, *, folder: str) -> None:
    for i, sample in enumerate(samples):
        image.imsave(
            (f"{folder}/sample_{i}.{IMAGE_FORMAT}"),
            (sample.numpy()),
            cmap="gray",
        )


def time_series_plot(
    x: np.ndarray, *, path: str, features_names: list, color: str
) -> None:
    k = len(features_names)
    aspect_ratio = 2
    h = 4
    fig, axes = plt.subplots(k, figsize=(h * aspect_ratio, h), sharex=True)
    for i, name in enumerate(features_names):
        axes[i].plot(x[:, :, i].T, alpha=0.5, color=color)
        axes[i].set_ylabel(name)
    plt.xlabel("t")
    fig.savefig(path)
    plt.close()


def time_series_plots(
    samples: np.ndarray,
    *,
    folder: str,
    n_plots: int,
    series_per_plot: int,
    features_names: list,
    color: str = "orange",
) -> None:
    assert samples.shape[-1] == len(features_names)

    for i in tqdm.tqdm(range(n_plots), desc="Generating time-series plots"):
        start = i * series_per_plot
        end = min(start + series_per_plot, len(samples))
        time_series_plot(
            x=samples[start:end],
            features_names=features_names,
            path=f"{folder}/{i}.{IMAGE_FORMAT}",
            color=color,
        )
    plt.close()


def summary_report(path: Path):
    stats = load_json(path / STATS_NAME)
    summary_stats = {}

    l1_div = {}
    if "numerical_stats" in stats["columnwise_comparison"]:
        l1_div.update(
            stats["columnwise_comparison"]["numerical_stats"]["l1_divergence"]
        )
    if "categorical_stats" in stats["columnwise_comparison"]:
        l1_div.update(
            stats["columnwise_comparison"]["categorical_stats"]["l1_histogram_distance"]
        )
    if "l1_divergence" in stats["columnwise_comparison"]:
        l1_div.update(stats["columnwise_comparison"]["l1_divergence"])

    summary_stats["l1_divergence"] = l1_div

    if "global_comparison" in stats:
        if ("numerical_stats" in stats["columnwise_comparison"]) and (
            "correlation matrix similarity" in stats["global_comparison"]
        ):
            summary_stats["correlations_l1"] = stats["global_comparison"][
                "numerical_stats"
            ]["correlation matrix similarity"]["average l1"]

        if (
            "categorical_stats" in stats["columnwise_comparison"]
            and "NMI matrix similarity" in stats["global_comparison"]
        ):
            summary_stats["NMI_l1"] = stats["global_comparison"]["categorical_stats"][
                "NMI matrix similarity"
            ]["average l1"]

        if (
            "correlations_l1" in stats["global_comparison"]
            and "correlation matrix similarity" in stats["global_comparison"]
        ):
            summary_stats["correlations_l1"] = stats["global_comparison"][
                "correlation matrix similarity"
            ]["average l1"]

    l1 = pd.Series(summary_stats["l1_divergence"])

    summary_stats["l1_divergence_median"] = l1.median()
    summary_stats["l1_divergence_mean"] = l1.mean()
    summary_stats["l1_divergence_max"] = l1.max()

    # TODO: for debug
    print(
        "\033[91m"
        + f"l1_divergence_max: {summary_stats['l1_divergence_max']:3g}"
        + "\033[0m"
    )

    store_json(summary_stats, file=path / Path(SUMMARY_STATS_NAME))


def discriminative_analysis(df_original, df_generated, path):
    metrics = xgboost_discriminator_metrics_with_kfold(df_original, df_generated)
    print("\033[91m Discriminator AUC:" + str(metrics["auc"]) + "\033[0m")
    store_json(metrics, file=path / "discriminative_analysis.json")

    with edit_json(path / Path(SUMMARY_STATS_NAME)) as summary:
        summary["XGBoost AUC"] = metrics["auc"]


def compare_dataframes(
    *, df_generated: pd.DataFrame, df_original: pd.DataFrame, path: Path
):

    dfs = {"generated": df_generated, "original": df_original}

    # Uniforming NaNs
    for c in df_generated.columns:
        if df_generated[c].dtype in (str, "O", bool, "category"):
            for df in dfs.values():
                df[c] = df[c].astype("str").replace(["nan", "<NA>"], "[M]")

    # with animated_message("Computing Statistics..."):
    statistics_comparison(df=dfs, file=path / Path(STATS_NAME))
    summary_report(path)

    with animated_message("Discriminative_analysis..."):
        discriminative_analysis(df_original, df_generated, path)

    with animated_message("Plotting..."):
        for comparison in (
            correlations_comparison,
            mutual_information_comparison,  # TODO: not really efficient NMI computed two times
            histograms_comparison,
        ):
            comparison(
                df_generated=dfs["generated"], df_train=dfs["original"], path=path
            )


def store_losses(folder: Path, train_losses, validation_losses=None):

    losses = {"train": train_losses}
    if validation_losses is not None:
        losses["validation"] = validation_losses

    d = {}
    for subset, losses in losses.items():
        d[subset] = {
            "last": losses[-1],
            "min": min(losses),
            "losses": list(losses),
        }

    store_json(d, file=folder / "losses.json")


def losses_plot(folder: Path, train_losses, validation_losses=None):
    plt.plot(
        np.arange(1, len(train_losses) + 1), train_losses, alpha=0.8, label="train"
    )
    if validation_losses is not None:
        plt.plot(
            np.arange(1, len(validation_losses) + 1),
            validation_losses,
            alpha=0.8,
            label="validation",
        )
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(folder / f"losses.{IMAGE_FORMAT}")
    plt.close()
