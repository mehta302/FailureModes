import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shap
from seaborn import kdeplot

from loader import load_eval, load_data, load_split


def plot_train_val_metrics(obj) -> None:
    """
    Plot the training and validation metrics.

    Parameters:
        Eval Metrics Object

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the training and validation losses
    ax.plot(obj["Train"]["rmse"], label="Train Loss")
    ax.plot(obj["Test"]["rmse"], label="Val Loss")

    ax.set_xlabel("Boost Rounds")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.suptitle("Train/Val Metrics")
    fig.tight_layout()

    plt.show()


def qq_plot(sample: np.ndarray, distribution: np.ndarray, viz=True) -> int:
    if viz:
        plt.figure(figsize=(8, 6))
        plt.scatter(distribution, sample, alpha=0.7, label="Predicted vs Observed")

        # Plot the perfect prediction line (y = x)
        min_val = min(np.min(distribution), np.min(sample))
        max_val = max(np.max(distribution), np.max(sample))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

        # Plot D10% bounds: y = 0.9x and y = 1.1x
        # x_vals = np.linspace(min_val, max_val, 500)
        # plt.plot(x_vals, 0.9 * x_vals, 'r--', label="10% Lower Bound")
        # plt.plot(x_vals, 1.1 * x_vals, 'r--', label="10% Upper Bound")

    # Compute D10% accuracy
    abs_errors = np.abs(sample - distribution)
    tolerance = 0.10 * np.abs(distribution)
    within_10_percent = abs_errors < tolerance
    d10_accuracy = 100 * np.sum(within_10_percent) / len(distribution)

    cov_div = sample / distribution
    coe_var = np.std(cov_div) / np.mean(cov_div)
    print(f"Coefficient of Variability: {np.std(cov_div) / np.mean(cov_div)}")
    # print(f"Covariance: {np.cov(sample, distribution)}")

    # Print D10% accuracy
    print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")
    # NOTE: RETURNING HERE

    if viz:
        plt.xlabel("Test Data Distribution (kN)")
        plt.ylabel("Sampled Data Distribution (kN)")
        plt.title("Decision Tree Model vs Chemofast Data (±10% Range)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return coe_var


def explainer(model, X_test) -> None:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.waterfall_plot(shap_values[0])


def quantile_regression(X_train, y_train, X_test, y_test) -> None:
    alpha = np.array([0.25, 0.5, 0.75])
    evals_result = {}

    Xy = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=True)
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy, enable_categorical=True)

    booster = xgb.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": alpha,
            "learning_rate": 0.2,
            "max_depth": 5
        },
        Xy,
        num_boost_round=32,
        early_stopping_rounds=5,
        evals=[(Xy, "Train"), (Xy_test, "Test")],
        evals_result=evals_result
    )

    print(np.isin(X_train["hef"], [0, 1]).any())
    xgb.plot_tree(booster)
    plt.show()

    # explainer = shap.Explainer(booster)
    # shap_values = explainer(X_test)
    # print(shap_values)
    # shap.waterfall_plot(shap_values[0, 0])

    # print(evals_result)

    # xgb.plot_tree(booster)
    # plt.show()
    # y_pred = booster.inplace_predict(X_test)
    # qq_plot(y_pred[:, 1], y_test)


def mse_regression(X_train, y_train, X_test, y_test) -> None:
    alpha = np.array([0.25, 0.5, 0.75])
    evals_result = {}

    Xy = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=True)
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy, enable_categorical=True)

    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "quantile_alpha": alpha,
            "learning_rate": 0.05,
            "max_depth": 3,
            "max_cat_to_onehot": 10,
            "min_child_weight": 2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42
        },
        Xy,
        num_boost_round=500,
        early_stopping_rounds=20,
        evals=[(Xy, "Train"), (Xy_test, "Test")],
        evals_result=evals_result
    )

    # xgb.plot_tree(booster)
    # plt.show()

    # explainer(booster, X_test)

    print(evals_result)

    y_pred = booster.inplace_predict(X_test)
    # kdeplot((y_test - y_pred), fill=True)
    # plt.axvline(0, color='red', linestyle='--')
    # plt.title("Residual KDE")
    # plt.show()
    qq_plot(y_pred, y_test)
    # plot_train_val_metrics(evals_result)


# def strat_kf(X, y, n_splits=5, shuffle=True, random_state=42, idx=(0, 1)) -> list[tuple[list[int], list[int]]]:
#     """
#     Stratified K-Fold cross-validator for regression tasks.
#
#     Stratifies based on categorical features in X, ensuring that each fold
#     maintains similar proportions of categorical feature combinations.
#
#     Parameters:
#     -----------
#     X : array-like of shape (n_samples, n_features)
#         Input features (assumed to be categorical)
#     y : array-like of shape (n_samples,)
#         Target values (continuous for regression)
#     n_splits : int, default=3
#         Number of folds
#     shuffle : bool, default=True
#         Whether to shuffle the data before splitting
#     random_state : int, default=42
#         Random state for reproducibility
#
#     Returns:
#     --------
#     list[tuple[list[int], list[int]]]
#         List of (train_indices, test_indices) tuples for each fold
#     """
#     np.random.seed(random_state)
#     X = np.array(X)
#     y = np.array(y)
#     n_samples = len(X)
#
#     # Create stratification groups based on categorical feature combinations
#     # Convert each row to a tuple for hashing
#     strata = [tuple(row[list(idx)]) for row in X]
#
#     # Group indices by their strata
#     strata_groups = {}
#     for idx, stratum in enumerate(strata):
#         if stratum not in strata_groups:
#             strata_groups[stratum] = []
#         strata_groups[stratum].append(idx)
#
#     # Shuffle indices within each stratum if requested
#     if shuffle:
#         for stratum in strata_groups:
#             np.random.shuffle(strata_groups[stratum])
#
#     # Initialize fold assignments
#     fold_assignments = np.full(n_samples, -1, dtype=int)
#
#     # Distribute each stratum across folds as evenly as possible
#     for stratum, indices in strata_groups.items():
#         n_stratum_samples = len(indices)
#
#         # Calculate base samples per fold and remainder
#         base_per_fold = n_stratum_samples // n_splits
#         remainder = n_stratum_samples % n_splits
#
#         # Assign samples to folds
#         idx_start = 0
#         for fold in range(n_splits):
#             # Some folds get one extra sample if there's a remainder
#             fold_size = base_per_fold + (1 if fold < remainder else 0)
#             fold_indices = indices[idx_start:idx_start + fold_size]
#
#             for idx in fold_indices:
#                 fold_assignments[idx] = fold
#
#             idx_start += fold_size
#
#     # Create train/test splits for each fold
#     splits = []
#     for fold in range(n_splits):
#         test_indices = [i for i in range(n_samples) if fold_assignments[i] == fold]
#         train_indices = [i for i in range(n_samples) if fold_assignments[i] != fold]
#         splits.append((train_indices, test_indices))
#
#     return splits


def cv_xgb(X, y, params) -> tuple[int, list]:
    # Create xgb params from params list
    xgb_params = {}
    for key, values, index in params:
        # Base Case
        if index == len(values):
            return 100, []

        xgb_params[key] = values[index]

    # Train model
    # Assumed constants
    xgb_params["quantile_alpha"] = np.array([0.25, 0.5, 0.75])
    xgb_params["tree_method"] = "hist"
    xgb_params["max_cat_to_onehot"] = 10
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    max_acc, max_params = 0, params

    # Test with cv
    for train_index, test_index in kf.split(X, y):
        X_test, y_test = X[test_index], y[test_index]
        Xy = xgb.QuantileDMatrix(X[train_index], y[train_index], enable_categorical=True)
        Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy, enable_categorical=True)

        booster = xgb.train(
            xgb_params,
            Xy,
            num_boost_round=500,
            early_stopping_rounds=20,
            evals=[(Xy, "Train"), (Xy_test, "Test")]
        )

        y_pred = booster.inplace_predict(X_test)

        max_acc += qq_plot(y_pred[:, 1] if xgb_params["objective"] == "reg:quantileerror" else y_pred, y_test, viz=False)
    max_acc /= 5

    # Return compare avg recursive calculate
    for i in range(len(params)):
        params[i][2] += 1
        new_acc, new_params = cv_xgb(X, y, params)
        params[i][2] -= 1
        if new_acc < max_acc:
            max_acc = new_acc
            max_params = [row[:] for row in new_params]

    return max_acc, [row[:] for row in max_params]


if __name__ == "__main__":
    # X, Y = load_eval()
    X, Y = load_data("Database")

    X["d"] = X["d"].astype("category")
    X["hef"] = X["hef"].astype("category")
    # X["Anchor diameter"] = X["Anchor diameter"].astype("category")
    # X["Embedment depth"] = X["Embedment depth"].astype("category")
    X["tau"] = X["tau"].astype(float)
    X["fc"] = X["fc"].astype("category")

    # X_train, X_test, Y_train, Y_test = load_split(sheet_name="Database1.5")
    # X = pd.concat([X_train, X_test])
    # Y = pd.concat([Y_train, Y_test])

    # _X, _Y = X.to_numpy(), Y.to_numpy()

    # train_idx, test_idx = strat_kf(_X, _Y)[0]

    # X_train = _X[train_idx]
    # X_test = _X[test_idx]
    # Y_train = _Y[train_idx]
    # Y_test = _Y[test_idx]

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42
    # )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # print(cv_xgb(X.to_numpy(), Y.to_numpy().astype(float), [
    #     ["objective", ["reg:squarederror", "reg:quantileerror"], 0],
    #     ["learning_rate", [0.01, 0.05, 0.1], 0],
    #     ["max_depth", [2, 3, 4], 0],
    # ]))

    # print(cv_xgb(X.to_numpy(), Y.to_numpy().astype(float), [
    #     ["objective", ["reg:squarederror"], 0],
    #     ["learning_rate", [0.01], 0],
    #     ["max_depth", [4, 5, 6], 0],
    # ]))

    # print(cv_xgb(X.to_numpy(), Y.to_numpy().astype(float), [
    #     ["objective", ["reg:squarederror"], 0],
    #     ["learning_rate", [0.01], 0],
    #     ["max_depth", [4], 0],
    #     ["subsample", [0.5, 0.8, 1.0], 0],
    #     ["min_child_weight", [1, 2, 3], 0]
    # ]))

    # print(cv_xgb(X.to_numpy(), Y.to_numpy().astype(float), [
    #     ["objective", ["reg:squarederror"], 0],
    #     ["learning_rate", [0.01], 0],
    #     ["max_depth", [4], 0],
    #     ["subsample", [0.8], 0],
    #     ["min_child_weight", [2], 0],
    #     ["colsample_bytree", [0.8, 0.9, 1.0], 0],
    #     ["reg_alpha", [0, 1], 0],
    #     ["reg_lambda", [0, 1], 0],
    # ]))

    # print(cv_xgb(X.to_numpy(), Y.to_numpy().astype(float), [
    #     ["objective", ["reg:squarederror"], 0],
    #     ["learning_rate", [0.1], 0],
    #     ["max_depth", [2], 0],
    #     ["subsample", [1.0], 0],
    #     ["min_child_weight", [1], 0],
    #     ["colsample_bytree", [1.0], 0],
    #     ["reg_alpha", [0, 0.1, 0.5], 0],
    #     ["reg_lambda", [1, 1.1, 1.5], 0],
    # ]))

    mse_regression(
    # quantile_regression(
        X_train,
        Y_train.to_numpy().astype(float),
        X_test,
        Y_test.to_numpy().astype(float)
        # X, Y, X, Y
    )

    # mse_regression(
    # quantile_regression(
    #     X_train,
    #     Y_train.astype(float),
    #     X_test,
    #     Y_test.astype(float)
    # )
