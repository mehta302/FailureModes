from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import shap

from loader import load_data


def qq_plot(sample: np.ndarray, distribution: np.ndarray) -> None:
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

    plt.xlabel("Test Data Distribution (kN)")
    plt.ylabel("Sampled Data Distribution (kN)")
    plt.title("Regression Model vs Chemofast Data (±10% Range)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y = load_data(sheet_name="Database3")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    lrm = LinearRegression()
    lrm.fit(X_train.to_numpy().astype(float), Y_train.to_numpy().astype(float))
    print(lrm.coef_)
    print(lrm.intercept_)

    # explainer = shap.explainers.Linear(lrm, X_test)
    # shap_values = explainer(X_test)
    # shap.waterfall_plot(shap_values[0])

    predictions = lrm.predict(X_test.to_numpy().astype(float))

    qq_plot(predictions, Y_test.to_numpy().astype(float))
