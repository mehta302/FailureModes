import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from seaborn import kdeplot

from loader import rload_data
from saver import rsave_data


class MLP(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        seed: int = 42
    ) -> None:
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.seed = seed
        torch.manual_seed(seed)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size1),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size1, hidden_size2),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size2, hidden_size3),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size3, 1)
        )
        self.model.double()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.model(X)
        return output

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        alpha: float
    ) -> None:

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), alpha)

        train_losses = []
        train_accs = []

        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            # Prepare the batches for training
            for X_batch, y_batch in zip(X_train.split(batch_size),
                                        y_train.split(batch_size)):
                # Set the model to train mode
                self.model.train()

                # Zero the gradients for this iteration
                optimizer.zero_grad()

                # Forward pass through the network
                output = self.forward(X_batch)

                # Calculate the loss
                loss = loss_fn(output, y_batch)

                # Backpropagation
                loss.backward()

                # Update the parameters
                optimizer.step()

            # Set the model to eval mode
            self.model.eval()
            # Use torch.no_grad() to prevent tracking history (and using memory)
            # since backpropagation is not needed for validation
            with torch.no_grad():
                train_losses.append(loss_fn(self.forward(X_train), y_train).item())
                train_accs.append(self.accuracy(X_train, y_train))

                val_losses.append(loss_fn(self.forward(X_val), y_val).item())
                val_accs.append(self.accuracy(X_val, y_val))

            # Print the loss and accuracy for every 5th epoch
            if (epoch+1) % 5 == 0:
                print(f"[{epoch+1} / {epochs}] | Train Loss: {train_losses[-1]:.5f} | Train Accuracy: {train_accs[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Accuracy: {val_accs[-1]:.5f}")

        self.train_val_metrics = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.forward(X)

    def plot_train_val_metrics(self) -> tuple[Figure, np.ndarray]:
        """
        Plot the training and validation metrics.

        Parameters:
            None

        Returns:
            Tuple[Figure, np.ndarray]: A tuple containing the matplotlib
                Figure and Axes objects.
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the training and validation losses
        ax[0].plot(self.train_val_metrics["train_losses"], label="Train Loss")
        ax[0].plot(self.train_val_metrics["val_losses"], label="Val Loss")

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot the training and validation accuracies
        ax[1].plot(self.train_val_metrics["train_accs"], label="Train Accuracy")
        ax[1].plot(self.train_val_metrics["val_accs"], label="Val Accuracy")

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Train/Val Metrics")
        fig.tight_layout()

        plt.savefig("mlp_train_val_metrics.png")

        return fig, ax

    def accuracy(self, X: torch.Tensor, y: torch.Tensor, viz=False) -> float:
        """
        Calculate the accuracy of the neural network on the input data.

        Parameters:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The true class labels.

        Returns:
            float: The accuracy of the neural network.
        """
        assert X.shape[0] == y.shape[0]
        y_pred = self.predict(X).numpy().flatten()
        # if viz:
        #     y_pred = np.exp(y_pred)
        y = y.numpy().flatten()

        # if viz:
        #     kdeplot((y - y_pred), fill=True)
        #     plt.axvline(0, color='red', linestyle='--')
        #     plt.title("Residual KDE (ANN)")
        #     plt.show()
        #
        #     return 0

        if viz:
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, alpha=0.7, label="Predicted vs Observed")
            # Plot the perfect prediction line (y = x)
            min_val = min(np.min(y), np.min(y_pred))
            max_val = max(np.max(y), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

            # Plot D10% bounds: y = 0.9x and y = 1.1x
            # x_vals = np.linspace(min_val, max_val, 500)
            # plt.plot(x_vals, 0.9 * x_vals, 'r--', label="10% Lower Bound")
            # plt.plot(x_vals, 1.1 * x_vals, 'r--', label="10% Upper Bound")

        # Compute D10% accuracy
        abs_errors = np.abs(y_pred - y)
        tolerance = 0.10 * np.abs(y)
        within_10_percent = abs_errors < tolerance
        d10_accuracy = np.sum(within_10_percent) / y.size

        coe_det = 1 - (np.sum(np.square(y_pred - y)) / np.sum(np.square(y - np.mean(y))))

        cov_div = y_pred / y
        coe_var = np.std(cov_div) / np.mean(cov_div)
        print(f"Coefficient of Variability: {coe_var}")

        print(f"Coefficient of Determinance: {coe_det}")

        # Print D10% accuracy
        # print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")

        if viz:
            # rsave_data(y_pred, "Artificial Neural Network")
            plt.xlabel("True Data Distribution (kN)")
            plt.ylabel("Predicted Data Distribution (kN)")
            plt.title("Artificial Neural Network Model")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return d10_accuracy


if __name__ == "__main__":
    X, Y = rload_data()
    Y = Y["Peak load-FE (kN) or Exp(kN)"]

    # X_train = pd.concat([X[46:93], X[::100]]).astype(float).to_numpy()
    # Y_train = pd.concat([Y[46:93], Y[::100]]).astype(float).to_numpy()
    # X_val = X[46:93].astype(float).to_numpy()
    # Y_val = Y[46:93].astype(float).to_numpy()
    # X_test = X[:46].astype(float).to_numpy()
    # Y_test = Y[:46].astype(float).to_numpy()
    # X_train = X[139:].astype(float).to_numpy()
    # Y_train = Y[139:].astype(float).to_numpy()
    # X_val = X[93:139].astype(float).to_numpy()
    # Y_val = Y[93:139].astype(float).to_numpy()
    # X_train, X_val, Y_train, Y_val = train_test_split(
    #     X[91:].to_numpy().astype(float),
    #     Y[91:].to_numpy().astype(float),
    #     test_size=0.2,
    #     random_state=42
    # )
    X_train = X[91:].astype(float).to_numpy()
    Y_train = Y[91:].astype(float).to_numpy()
    X_test = X[:91].astype(float).to_numpy()
    Y_test = Y[:91].astype(float).to_numpy()

    indices = np.random.permutation(len(Y_train))
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X.to_numpy().astype(float),
    #     Y.to_numpy().astype(float),
    #     test_size=0.2,
    #     random_state=42
    # )
    # X_train, X_val, Y_train, Y_val = train_test_split(
    #     X_train, Y_train, test_size=0.1, random_state=42
    # )

    print("[Data Loaded]")
    print(X_train.shape)

    Model = MLP(input_size=4, hidden_size1=64, hidden_size2=32, hidden_size3=16).to("cpu")

    print("[Model Loaded]")

    Model.train(
        X_train=torch.from_numpy(X_train).double(),
        y_train=torch.from_numpy(Y_train).double().reshape(-1, 1),
        X_val=torch.from_numpy(X_test).double(),
        y_val=torch.from_numpy(Y_test).double().reshape(-1, 1),
        epochs=205, #205,
        batch_size=30, #30,
        alpha=0.01
    )

    print("[Model Trained]")

    Model.plot_train_val_metrics()

    # print(Model.accuracy(torch.from_numpy(X_test).double(), torch.from_numpy(Y_test).double(), viz=True))
