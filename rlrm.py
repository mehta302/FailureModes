from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from seaborn import kdeplot

from loader import rload_data
from saver import rsave_data


def qq_plot(sample: np.ndarray, distribution: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(distribution, sample, alpha=0.7, label="True vs Pred")

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

    print(f"Coefficient of Determinance: {coe_det}")

    # Print D10% accuracy
    print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")

    plt.xlabel("True Data Distribution (kN)")
    plt.ylabel("Predicted Data Distribution (kN)")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y = rload_data()

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42
    # )
    X_train = X[91:]
    Y_train = Y[91:]
    X_test = X[:91]
    Y_test = Y[:91]
    # X_train = X[46:184]
    # Y_train = Y[46:184]
    # X_test = X[:46]
    # Y_test = Y[:46]
    # X_train = pd.concat([X[46:93], X[::100]])
    # Y_train = pd.concat([Y[46:93], Y[::100]])
    # X_test = X[:46]
    # Y_test = Y[:46]
    # X_train = X[:]
    # Y_train = Y[:]
    # X_test = X[:]
    # Y_test = Y[:]

    lrm = LinearRegression()
    # lrm.fit(X_train.to_numpy().astype(float), Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float))
    lrm.fit(np.log(X_train.to_numpy().astype(float)), np.log(Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float)))
    print(lrm.coef_)
    print(lrm.intercept_)

    # predictions = lrm.predict(X_test.to_numpy().astype(float))
    predictions = lrm.predict(np.log(X_test.to_numpy().astype(float)))

    # qq_plot(predictions, Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float))
    qq_plot(np.exp(predictions), Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float))

    # kdeplot((Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float) - np.exp(predictions)), fill=True)
    # plt.axvline(0, color='red', linestyle='--')
    # plt.title("Residual KDE")
    # plt.show()

    rsave_data(np.exp(predictions), "ML Linear Regression")
