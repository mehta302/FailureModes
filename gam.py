from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, f
from matplotlib import pyplot as plt
import numpy as np

from loader import load_eval


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


# def qq_plot(sample: np.ndarray, distribution: np.ndarray, interval: np.ndarray,
#             deviations: int = 2
#             ) -> None:
#     plt.scatter(distribution, sample)
#     # plt.plot(np.array([0, 100]), np.array([0, 100]))
#     plt.plot(distribution, distribution)
#     # plt.errorbar(distribution, sample, yerr=interval, fmt="o")
#     plt.xlabel("Test Data Distribution (kN)")
#     plt.ylabel("Sampled Data Distribution (kN)")
#     plt.title(f"GAM Model vs Chemofast Data ({deviations}*SE)")
#     plt.show()

def qq_plot(sample: np.ndarray, distribution: np.ndarray, std: np.ndarray,
            deviations: int = 2
            ) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(distribution, sample, alpha=0.7, label="Predicted vs Observed")

    # Plot the perfect prediction line (y = x)
    min_val = min(np.min(distribution), np.min(sample))
    max_val = max(np.max(distribution), np.max(sample))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

    # Plot D10% bounds: y = 0.9x and y = 1.1x
    x_vals = np.linspace(min_val, max_val, 500)
    plt.plot(x_vals, 0.9 * x_vals, 'r--', label="10% Lower Bound")
    plt.plot(x_vals, 1.1 * x_vals, 'r--', label="10% Upper Bound")

    # Compute D10% accuracy
    abs_errors = np.abs(sample - distribution)
    tolerance = 0.10 * np.abs(distribution)
    within_10_percent = abs_errors < tolerance
    d10_accuracy = 100 * np.sum(within_10_percent) / len(distribution)

    # Print D10% accuracy
    print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")

    # Optionally, show std-deviation error bars (commented out here)
    # plt.errorbar(distribution, sample, yerr=(2 * deviations * std), fmt="o", alpha=0.3)

    plt.xlabel("Test Data Distribution (kN)")
    plt.ylabel("Sampled Data Distribution (kN)")
    plt.title("GAM Model vs Chemofast Data (±10% Range)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y = load_eval()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    gam = LinearGAM(f(0) + f(1) + s(2, n_splines=5) + s(3, n_splines=4)).gridsearch(
        X_train.to_numpy().astype(float),
        Y_train.to_numpy().astype(float)
    )
    interval = gam.prediction_intervals(X_test.to_numpy().astype(float))
    pred = gam.predict(X_test.to_numpy().astype(float))
    y_test = Y_test.to_numpy().astype(float)
    # gam.summary()
    print(f"[Interval: {interval}]")
    print(f"[Predictions: {pred}]")
    print(f"[Test: {y_test}]")
    print(f"[Accuracy of 95% reference: {accuracy(interval, y_test)}]")
    print(f"[Avg Var: {avg_var(interval)}]")

    std = []
    for i in range(interval.shape[0]):
        std.append(interval[i, 1] - interval[i, 0])
    std = np.array(std)

    qq_plot(pred, y_test, std)

    # plt.figure()
    # fig, axs = plt.subplots(1, 4)
    #
    # XX = gam.generate_X_grid(0)
    # XX[:, 1] = gam.generate_X_grid(1)[:, 1]
    # XX[:, 2] = gam.generate_X_grid(2)[:, 2]
    # XX[:, 3] = gam.generate_X_grid(3)[:, 3]
    #
    # cX_test = X_test.to_numpy().astype(float)
    # cY_test = Y_test.to_numpy().astype(float)
    # cY_test -= np.mean(cY_test)
    #
    # for i, ax in enumerate(axs):
    #     ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    #     ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    #     ax.scatter(cX_test[:, i], cY_test, facecolor='grey')
    #
    # plt.show()
