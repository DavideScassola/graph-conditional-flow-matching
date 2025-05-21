import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

PRECISION_DIGITS = 3


def xgboost_discriminator_metrics(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    test_proportion: float = 0.2,
    random_state=1234,
) -> dict:

    # Add labels
    real_data["label"] = 1
    generated_data["label"] = 0

    # Combine the data
    data = pd.concat([real_data, generated_data])

    # Convert object columns to categorical
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype("category")

    # Split features and labels
    X = data.drop(columns=["label"])
    y = data["label"]

    # drop_labels from original dataset # TODO: not ideal
    del real_data["label"]
    del generated_data["label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_proportion, random_state=random_state
    )

    # Train the XGBoost classifier with a seed
    model = xgb.XGBClassifier(
        eval_metric="logloss", enable_categorical=True, random_state=random_state
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cross_entropy = log_loss(y_test, y_pred_proba)

    # Features importance features
    importance = np.round(model.feature_importances_, PRECISION_DIGITS)
    feature_names = X.columns
    feature_importance = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "accuracy": np.round(accuracy, PRECISION_DIGITS),
        "auc": np.round(auc, PRECISION_DIGITS),
        "cross_entropy": np.round(cross_entropy, PRECISION_DIGITS),
        "feature_importance": feature_importance,
    }


def xgboost_discriminator_metrics_with_kfold(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    n_splits: int = 5,
    random_state=1234,
) -> dict:
    # Add labels
    real_data["label"] = 1
    generated_data["label"] = 0

    # Combine the data
    data = pd.concat([real_data, generated_data])

    # Convert object columns to categorical
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype("category")

    # Split features and labels
    X = data.drop(columns=["label"])
    y = data["label"]

    # Drop labels from original datasets
    del real_data["label"]
    del generated_data["label"]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Metrics to track
    accuracies = []
    aucs = []
    cross_entropies = []

    # Perform Stratified K-Fold Cross-Validation with tqdm progress bar
    for train_index, test_index in tqdm(
        skf.split(X, y), total=n_splits, desc="Cross-Validation Progress"
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the XGBoost classifier
        model = xgb.XGBClassifier(
            eval_metric="logloss", enable_categorical=True, random_state=random_state
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_pred_proba))
        cross_entropies.append(log_loss(y_test, y_pred_proba))

    # Compute average and standard deviation for metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    avg_cross_entropy = np.mean(cross_entropies)
    std_cross_entropy = np.std(cross_entropies)

    # Print metrics in avg ± err notation
    # print(f"Accuracy: {avg_accuracy:.3g} ± {std_accuracy:.3g}")
    # print(f"AUC: {avg_auc:.3g} ± {std_auc:.3g}")
    # rint(f"Cross-Entropy: {avg_cross_entropy:.3g} ± {std_cross_entropy:.3g}")

    # Features importance features
    importance = np.round(model.feature_importances_, PRECISION_DIGITS)
    feature_names = X.columns
    feature_importance = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "accuracy": f"{avg_accuracy:.3g} ± {std_accuracy:.3g}",
        "auc": f"{avg_auc:.3g} ± {std_auc:.3g}",
        "cross_entropy": f"{avg_cross_entropy:.3g} ± {std_cross_entropy:.3g}",
        "feature_importance": feature_importance,
    }
