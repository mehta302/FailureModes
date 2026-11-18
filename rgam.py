from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from seaborn import kdeplot

from loader import rload_data
from saver import rsave_data


def accuracy(interval: np.ndarray, y_test: np.ndarray) -> float:
    correct = 0
    for i in range(y_test.size):
        if y_test[i] >= interval[i, 0] and y_test[i] <= interval[i, 1]:
            correct += 1

    return correct / y_test.size


def avg_var(interval: np.ndarray) -> float:
    avg_95 = 0
    for i in range(interval.shape[0]):
        avg_95 += interval[i, 1] - interval[i, 0]
    avg_95 /= interval.shape[0]

    stddev = avg_95 / 4
    return (stddev ** 2)


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
    coe_det = 1 - (np.sum(np.square(sample - distribution)) / np.sum(np.square(distribution - np.mean(distribution))))

    # Compute D10% accuracy
    abs_errors = np.abs(sample - distribution)
    tolerance = 0.10 * np.abs(distribution)
    within_10_percent = abs_errors < tolerance
    d10_accuracy = 100 * np.sum(within_10_percent) / len(distribution)

    cov_div = sample / distribution
    coe_var = np.std(cov_div) / np.mean(cov_div)
    print(f"Coefficient of Variability: {coe_var}")

    print(f"Coefficient of Determinance: {coe_det}")

    # Print D10% accuracy
    print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")

    # Optionally, show std-deviation error bars (commented out here)
    # plt.errorbar(distribution, sample, yerr=(2 * deviations * std), fmt="o", alpha=0.3)

    plt.xlabel("True Data Distribution (kN)")
    plt.ylabel("Predicted Data Distribution (kN)")
    plt.title("Generalized Additive Model")
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

    # gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5)).gridsearch(
    #     X_train.to_numpy().astype(float),
    #     Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float)
    # )
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(
        np.log(X_train.to_numpy().astype(float)),
        np.log(Y_train["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float))
    )
    # interval = gam.prediction_intervals(X_test.to_numpy().astype(float))
    # pred = gam.predict(X_test.to_numpy().astype(float))
    pred = gam.predict(np.log(X_test.to_numpy().astype(float)))
    y_test = Y_test["Peak load-FE (kN) or Exp(kN)"].to_numpy().astype(float)

    # kdeplot((y_test - np.exp(pred)), fill=True)
    # plt.axvline(0, color='red', linestyle='--')
    # plt.title("Residual KDE (GAM)")
    # plt.show()
    # gam.summary()
    # print(f"[Interval: {interval}]")
    # print(f"[Predictions: {pred}]")
    # print(f"[Test: {y_test}]")
    # print(f"[Accuracy of 95% reference: {accuracy(interval, y_test)}]")
    # print(f"[Avg Var: {avg_var(interval)}]")

    # std = []
    # for i in range(interval.shape[0]):
    #     std.append(interval[i, 1] - interval[i, 0])
    # std = np.array(std)
    # plt.figure()

    # fig, axs = plt.subplots(1, 4)
    #
    # XX = gam.generate_X_grid(0)
    # XX[:, 1] = gam.generate_X_grid(1)[:, 1]
    # XX[:, 2] = gam.generate_X_grid(2)[:, 2]
    # XX[:, 3] = gam.generate_X_grid(3)[:, 3]
    #
    # cX_test = np.log(X_test.to_numpy().astype(float))
    # cY_test = np.log(y_test)
    # cY_test -= np.mean(cY_test)
    #
    # for i, ax in enumerate(axs):
    #     ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    #     # ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    #     ax.scatter(cX_test[:, i], cY_test, facecolor='grey')
    #     ax.title.set_text(X.columns[i])
    #
    # plt.show()

    qq_plot(np.exp(pred), y_test)

    rsave_data(np.exp(pred), "Generalized Additive Model")
