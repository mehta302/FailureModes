from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

from loader import load_eval


def accuracy(mu: np.ndarray, std: np.ndarray, y_test: np.ndarray) -> float:
    within_interval = (y_test >= (mu - (2 * std))) & (y_test <= (mu + (2 * std)))
    # within_interval = (y_test >= (mu - (std))) & (y_test <= (mu + (std)))
    correct = np.sum(within_interval)
    return correct / y_test.size


# def qq_plot(sample: np.ndarray, distribution: np.ndarray, std: np.ndarray,
#             deviations: int = 2
#             ) -> None:
#     plt.scatter(distribution, sample)
#     # plt.plot(distribution, distribution)
#     plt.plot(np.array([0, 100]), np.array([0, 100]))
#     # plt.errorbar(distribution, sample, yerr=(2 * deviations * std), fmt="o")
#     plt.xlabel("Test Data Distribution (kN)")
#     plt.ylabel("Sampled Data Distribution (kN)")
#     plt.title(f"GPR Model vs Chemofast Data ({deviations}*STD)")
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
    plt.title("GPR Model vs Chemofast Data (±10% Range)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y = load_eval()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    step = ExpSineSquared() * RBF()
    spread = RBF() * WhiteKernel()
    quad = DotProduct() * DotProduct()
    kernel = (2 * step) + spread + quad
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(
        X_train.to_numpy().astype(float),
        Y_train.to_numpy().astype(float)
    )

    y_test = Y_test.to_numpy().astype(float)
    mu, std = gp.predict(X_test.to_numpy().astype(float), return_std=True)
    qq_plot(mu, y_test, std)
    print(mu)
    print(std)
    print(np.mean(std ** 2))
    print(accuracy(mu, std, y_test))
