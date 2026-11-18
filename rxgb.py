import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from seaborn import kdeplot
import shap

from loader import rload_data
from saver import rsave_data


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

    coe_det = 1 - (np.sum(np.square(sample - distribution)) / np.sum(np.square(distribution - np.mean(distribution))))

    cov_div = sample / distribution
    coe_var = np.std(cov_div) / np.mean(cov_div)
    print(f"Coefficient of Variability: {coe_var}")
    # print(f"Covariance: {np.cov(sample, distribution)}")

    print(f"Coefficient of Determinance: {coe_det}")

    # Print D10% accuracy
    print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")
    # NOTE: RETURNING HERE

    if viz:
        plt.xlabel("True Data Distribution (kN)")
        plt.ylabel("Predicted Data Distribution (kN)")
        plt.title("XGBoost Model")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return d10_accuracy


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

    # xgb.plot_tree(booster)
    # plt.show()

    # explainer = shap.Explainer(booster)
    # shap_values = explainer(X_test)
    # print(shap_values)
    # shap.waterfall_plot(shap_values[0, 0])

    print(evals_result)

    # xgb.plot_tree(booster)
    # plt.show()
    y_pred = booster.inplace_predict(X_test)
    qq_plot(y_pred[:, 1], y_test)
    # rsave_data(y_pred, "Decision Tree")


def explainer(model, X_test) -> None:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values)


def mse_regression(X_train, y_train, X_test, y_test) -> None:
    alpha = np.array([0.25, 0.5, 0.75])
    evals_result = {}

    # Xy = xgb.QuantileDMatrix(np.log(X_train), np.log(y_train), enable_categorical=True)
    # Xy_test = xgb.QuantileDMatrix(np.log(X_test), np.log(y_test), ref=Xy, enable_categorical=True)
    Xy = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=True)
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy, enable_categorical=True)

    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "quantile_alpha": alpha,
            "learning_rate": 0.001,
            "max_depth": 5,
            "max_cat_to_onehot": 10,
            "min_child_weight": 10,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42
            # "objective": "reg:squarederror",
            # "tree_method": "hist",
            # "quantile_alpha": alpha,
            # "learning_rate": 0.01,
            # "max_depth": 4,
            # "max_cat_to_onehot": 10,
            # "min_child_weight": 1,
            # "subsample": 1.0,
            # "colsample_bytree": 1.0,
            # "gamma": 0,
            # "reg_alpha": 0,
            # "reg_lambda": 1,
            # "random_state": 42
        },
        Xy,
        num_boost_round=5000,
        early_stopping_rounds=200,
        evals=[(Xy, "Train"), (Xy_test, "Test")],
        evals_result=evals_result,
    )

    # xgb.plot_tree(booster)
    # plt.show()

    # explainer(booster, X_test)

    print(evals_result)

    y_pred = booster.inplace_predict(X_test)
    # y_pred = booster.inplace_predict(np.log(X_test))
    # kdeplot((y_test - y_pred), fill=True)
    # plt.axvline(0, color='red', linestyle='--')
    # plt.title("Residual KDE")
    # plt.show()
    # kdeplot((y_test - y_pred), fill=True)
    # plt.axvline(0, color='red', linestyle='--')
    # plt.title("Residual KDE (XGB)")
    # plt.show()

    qq_plot(y_pred, y_test)
    # qq_plot(np.exp(y_pred), y_test)
    # plot_train_val_metrics(evals_result)
    rsave_data(y_pred, "Decision Tree")


def cv_xgb(X_train, y_train, X_test, y_test, params) -> tuple[int, list]:
    # Create xgb params from params list
    xgb_params = {}
    for key, values, index in params:
        # Base Case
        if index == len(values):
            return 0, []

        xgb_params[key] = values[index]

    # Train model
    # Assumed constants
    xgb_params["quantile_alpha"] = np.array([0.25, 0.5, 0.75])
    xgb_params["tree_method"] = "hist"
    xgb_params["max_cat_to_onehot"] = 10
    max_acc, max_params = 0, params

    # Test
    Xy = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=True)
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy, enable_categorical=True)

    booster = xgb.train(
        xgb_params,
        Xy,
        num_boost_round=500,
        early_stopping_rounds=20,
        evals=[(Xy, "Train"), (Xy_test, "Test")]
    )

    y_pred = booster.inplace_predict(X_test)

    max_acc = qq_plot(y_pred[:, 1] if xgb_params["objective"] == "reg:quantileerror" else y_pred, y_test, viz=False)

    # Return compare avg recursive calculate
    for i in range(len(params)):
        params[i][2] += 1
        new_acc, new_params = cv_xgb(X_train, y_train, X_test, y_test, params)
        params[i][2] -= 1
        if new_acc > max_acc:
            max_acc = new_acc
            max_params = [row[:] for row in new_params]

    return max_acc, [row[:] for row in max_params]


if __name__ == "__main__":
    X, Y = rload_data()

    # X["Dia (mm)"] = X["Dia (mm)"].astype("category")
    # X["hef (mm)"] = X["hef (mm)"].astype("category")
    # X["Taub-Mortar (MPa)"] = X["Taub-Mortar (MPa)"].astype("category")
    # X["f'c (MPa)"] = X["f'c (MPa)"].astype("category")
    # X["Cmin"] = X["Cmin"].astype("category")
    # X["Cmax"] = X["Cmax"].astype("category")

    # X["Dia (mm)"] = X["Dia (mm)"].astype(float)
    X["hef (mm)"] = X["hef (mm)"].astype(float)
    X["f'c (MPa)"] = X["f'c (MPa)"].astype(float)
    # X["Taub-Mortar (MPa)"] = X["Taub-Mortar (MPa)"].astype(float)
    X["Cmin"] = X["Cmin"].astype(float)
    X["Cmax"] = X["Cmax"].astype(float)
    # X["C/D"] = X.apply(lambda row: (row[4] + row[5]) / (2 * row[0]))
    # X["H/D"] = X.apply(lambda row: row[1] / row[0])
    # X["Conf"] = X.apply(lambda row: sqrt(row[2]) * (row[4] + row[5]) / row[0])
    # X["Res"] = X.apply(lambda row: sqrt(row[2]) * row[4])
    # X["Inter1"] = X.apply(lambda row: row[3] * row[1])
    # X["Inter2"] = X.apply(lambda row: row[2] * (row[4] ** 2))

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42
    # )
    # X_train = pd.concat([X[46:91], X[::100]])
    # Y_train = pd.concat([Y[46:91], Y[::100]])
    # X_test = X[:46]
    # Y_test = Y[:46]
    X_train = X[91:]
    Y_train = Y[91:]
    X_test = X[:91]
    Y_test = Y[:91]

    print("Starting Regression...")

    # print(cv_xgb(
    #     X_train,
    #     Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     X_test,
    #     Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     [
    #         ["objective", ["reg:squarederror"], 0],
    #         ["learning_rate", [0.1, 0.2, 0.3], 0],
    #         ["max_depth", [2, 3, 4], 0],
    #     ]
    # ))
    # print(cv_xgb(
    #     X_train,
    #     Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     X_test,
    #     Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     [
    #         ["objective", ["reg:squarederror"], 0],
    #         ["learning_rate", [0.001, 0.005, 0.01], 0],
    #         ["max_depth", [6, 7, 8], 0],
    #     ]
    # ))
    # print(cv_xgb(
    #     X_train,
    #     Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     X_test,
    #     Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
    #     [
    #         ["objective", ["reg:squarederror"], 0],
    #         ["learning_rate", [0.005], 0],
    #         ["max_depth", [7], 0],
    #         ["subsample", [1.0], 0],
    #         ["min_child_weight", [1], 0],
    #         ["colsample_bytree", [1.0], 0],
    #         ["gamma", [0, 1], 0],
    #         ["reg_alpha", [1, 2], 0],
    #         ["reg_lambda", [1, 2], 0],
    #     ]
    # ))

    mse_regression(
        X_train,
        Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float),
        X_test,
        Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float)
    )
