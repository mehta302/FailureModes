import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from loader import load_excel


class MLPMCD(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        seed: int = 42
    ) -> None:

        super(MLPMCD, self).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.seed = seed
        torch.manual_seed(seed)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_size1, hidden_size2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_size2, 1)
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
            # if (epoch+1) % 5 == 0:
                # print(f"[{epoch+1} / {epochs}] | Train Loss: {train_losses[-1]:.5f} | Train Accuracy: {train_accs[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Accuracy: {val_accs[-1]:.5f}")

        self.train_val_metrics = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }

    def enable_dropout(self) -> None:

        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def predict(self, X: torch.Tensor, n_passes: int) -> tuple[torch.Tensor, torch.Tensor]:

        predictions = np.zeros((X.shape[0], n_passes))

        for i in range(n_passes):
            self.model.eval()
            self.enable_dropout()
            with torch.no_grad():
                output = self.forward(X)
                predictions[:, i] = output.squeeze().numpy()

        mu = np.mean(predictions, axis=1)
        sig = np.sqrt(np.var(predictions, axis=1))

        return mu, sig

    def accuracy(self, X: torch.Tensor, y: torch.Tensor, n_passes: int = 30) -> float:

        assert X.shape[0] == y.shape[0]
        mu, sig = self.predict(X, n_passes)
        y_np = y.squeeze().numpy()
        within_interval = (y_np >= (mu - (2 * sig))) & (y_np <= (mu + (2 * sig)))
        correct = np.sum(within_interval)
        acc = correct / X.shape[0]
        return acc

    def print_accuracy(self, X: torch.Tensor, y: torch.Tensor, n_passes: int = 30) -> None:

        assert X.shape[0] == y.shape[0]
        mu, sig = self.predict(X, n_passes)
        y_np = y.squeeze().numpy()
        within_interval = (y_np >= (mu - (2 * sig))) & (y_np <= (mu + (2 * sig)))
        correct = np.sum(within_interval)
        acc = correct / X.shape[0]
        print(f"[Avg Var: {np.mean(sig ** 2)} | Accuracy: {acc}] [H1: {self.hidden_size1} | H2: {self.hidden_size2}]")
        # plt.scatter(y, mu)
        # plt.plot(y, y)
        # plt.errorbar(y, mu, yerr=(2*sig), fmt="o")
        # plt.xlabel("Test Data Distribution (kN)")
        # plt.ylabel("Sampled Data Distribution (kN)")
        # plt.title("MCD Model vs Chemofast Data (2*STD)")
        # plt.show()
        plt.figure(figsize=(8, 6))
        plt.scatter(y, mu, alpha=0.7, label="Predicted vs Observed")

        # Plot the perfect prediction line (y = x)
        min_val = min(min(y), min(mu))
        max_val = max(max(y), max(mu))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

        # Plot D10% bounds: y = 0.9x and y = 1.1x
        x_vals = np.linspace(min_val, max_val, 500)
        plt.plot(x_vals, 0.9 * x_vals, 'r--', label="10% Lower Bound")
        plt.plot(x_vals, 1.1 * x_vals, 'r--', label="10% Upper Bound")

        # Compute D10% accuracy
        abs_errors = np.abs(mu - y.numpy())
        tolerance = 0.10 * np.abs(y.numpy())
        within_10_percent = abs_errors < tolerance
        d10_accuracy = 100 * np.sum(within_10_percent) / len(mu)

        # Print D10% accuracy
        print(f"D10% Accuracy: {d10_accuracy:.2f}% of predictions are within ±10% of observed values")

        # Optionally, show std-deviation error bars (commented out here)
        # plt.errorbar(distribution, sample, yerr=(2 * deviations * std), fmt="o", alpha=0.3)

        plt.xlabel("Test Data Distribution (kN)")
        plt.ylabel("Sampled Data Distribution (kN)")
        plt.title(f"MCD Model vs Chemofast Data (±10% Range)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_train_val_metrics(self) -> None:
        """
        Plot the training and validation metrics.

        Parameters:
            None

        Returns:
            None
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

        plt.show()

if __name__ == "__main__":
    X, y = load_excel()
    X_train, X_test, y_train, y_test = train_test_split(
        torch.from_numpy(X.to_numpy().astype(float)),
        torch.from_numpy(y.to_numpy().astype(float)),
        test_size=0.2,
        random_state=42
    )
    X_train = X_train.double()
    y_train = y_train.double()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print("[Data Loaded]")
    print(X_train)
    print(y_train)

    # for lr in range(1, 11):
    #     print(lr / 1000)
    # for h1 in range(16, 66, 2):
    #     for h2 in range(8, 33):
    #         Model = MLPMCD(input_size=4, hidden_size1=h1, hidden_size2=h2).to("cpu")
    #         print("[Model Loaded]")
    #         Model.train(
    #             X_train=X_train,
    #             y_train=y_train,
    #             X_val=X_val,
    #             y_val=y_val,
    #             epochs=100,
    #             batch_size=30,
    #             alpha=0.001
    #             alpha=(lr / 1000)
    #         )
    #         print("[Model Trained]")
    #         Model.print_accuracy(X_test, y_test)

    # Model = MLPMCD(input_size=4, hidden_size1=64, hidden_size2=27).to("cpu")
    Model = MLPMCD(input_size=4, hidden_size1=16, hidden_size2=9).to("cpu")
    print("[Model Loaded]")

    Model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=30,
        alpha=0.001
    )
    print("[Model Trained]")

    Model.print_accuracy(X_test, y_test)

    # test_accuracy = Model.accuracy(X_test, y_test)
    # print(f"[Test_accuracy: {test_accuracy}]")
    # Model.plot_train_val_metrics()
