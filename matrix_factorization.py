
"""M3-M4 - Matrix Factorization with SGD and Grid Search."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baselines import load_ratings, split_ratings


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_K = 20 #Latent factors
DEFAULT_REG = 0.1 #Lambda
DEFAULT_LEARNING_RATE = 0.01

EPOCHS = 50
SEED = 42

K_VALUES = [5, 10, 20, 30]
REGULARIZATIONS_VALUES = [0.05, 0.1, 0.2]
LEARNING_RATE_VALUES = [0.005, 0.01, 0.02]


# ==========================================================
# 1. MATRIX FACTORIZATION WITH SGD (M3)
# ==========================================================

class MatrixFactorizationSGD:

    def __init__(self, k=DEFAULT_K, learning_rate=DEFAULT_LEARNING_RATE,
                 reg=DEFAULT_REG, epochs=EPOCHS, seed=SEED):

        self.k = k
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def fit(self, train, val):
        """Learn user/item biases and latent factors using SGD."""

        rng = np.random.default_rng(self.seed)

        # Map user and movie IDs to matrix indices
        self.user_ids = pd.Index(train["user_id"].unique())
        self.movie_ids = pd.Index(train["movie_id"].unique())

        user_idx = self.user_ids.get_indexer(train["user_id"])
        movie_idx = self.movie_ids.get_indexer(train["movie_id"])
        ratings = train["rating"].to_numpy(dtype=float)

        # Initialization
        self.mu = ratings.mean()
        self.U = rng.normal(0, 0.1, size=(len(self.user_ids), self.k))
        self.V = rng.normal(0, 0.1, size=(len(self.movie_ids), self.k))

        self.user_bias = np.zeros(len(self.user_ids))
        self.movie_bias = np.zeros(len(self.movie_ids))

        self.history = []
        self.best_epoch = 0
        self.best_val_rmse = float("inf")

        # --------------------------------------------------
        # SGD Training
        # --------------------------------------------------

        for epoch in range(self.epochs):

            # Shuffle observed ratings each epoch
            for pos in rng.permutation(len(ratings)):

                u = user_idx[pos]
                i = movie_idx[pos]
                r = ratings[pos]

                # Previous factors
                p = self.U[u].copy()
                q = self.V[i].copy()

                # Prediction and error
                prediction = (
                    self.mu
                    + self.user_bias[u]
                    + self.movie_bias[i]
                    + p @ q
                )

                error = r - prediction

                # Update biases
                self.user_bias[u] += self.learning_rate * (
                    error - self.reg * self.user_bias[u]
                )

                self.movie_bias[i] += self.learning_rate * (
                    error - self.reg * self.movie_bias[i]
                )

                # Update latent factors
                self.U[u] += self.learning_rate * (
                    error * q - self.reg * p
                )

                self.V[i] += self.learning_rate * (
                    error * p - self.reg * q
                )

            # Track training and validation RMSE
            train_rmse = self.rmse(train)
            val_rmse = self.rmse(val)

            self.history.append({
                "Epoch": epoch + 1,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": val_rmse
            })

            # Save parameters at the best validation epoch
            if val_rmse < self.best_val_rmse:

                self.best_val_rmse = val_rmse
                self.best_epoch = epoch + 1

                best_params = (
                    self.U.copy(),
                    self.V.copy(),
                    self.user_bias.copy(),
                    self.movie_bias.copy()
                )

        # Restore the best model
        (
            self.U,
            self.V,
            self.user_bias,
            self.movie_bias
        ) = best_params

        return self

    def predict(self, data):
        """Predict ratings, handling unknown users and movies."""

        users = self.user_ids.get_indexer(data["user_id"])
        movies = self.movie_ids.get_indexer(data["movie_id"])

        pred = np.full(len(data), self.mu)

        known_users = users >= 0
        known_movies = movies >= 0
        both_known = known_users & known_movies

        # User and movie biases
        pred[known_users] += self.user_bias[users[known_users]]
        pred[known_movies] += self.movie_bias[movies[known_movies]]

        # Latent factor interaction
        pred[both_known] += np.sum(
            self.U[users[both_known]]
            * self.V[movies[both_known]],
            axis=1
        )

        return np.clip(pred, 1, 5)

    def rmse(self, data):
        """Compute RMSE on observed ratings."""

        y = data["rating"].to_numpy(dtype=float)
        pred = self.predict(data)

        return np.sqrt(np.mean((y - pred) ** 2))


# ==========================================================
# 2. GRID SEARCH (M4)
# ==========================================================

def grid_search(train, val,
                ranks=K_VALUES,
                regularizations=REGULARIZATIONS_VALUES,
                learning_rates=LEARNING_RATE_VALUES):
    """Select rank, regularization, learning rate and epoch."""

    results = []
    best_model = None
    best_rmse = float("inf")

    for k in ranks:
        for reg in regularizations:
            for lr in learning_rates:

                print(f"Training: k={k}, reg={reg}, lr={lr}")

                model = MatrixFactorizationSGD(
                    k=k,
                    reg=reg,
                    learning_rate=lr
                ).fit(train, val)

                results.append({
                    "Rank": k,
                    "Regularization": reg,
                    "Learning_Rate": lr,
                    "Best_Epoch": model.best_epoch,
                    "Validation_RMSE": model.best_val_rmse
                })

                if model.best_val_rmse < best_rmse:
                    best_rmse = model.best_val_rmse
                    best_model = model

    results = (
        pd.DataFrame(results)
        .sort_values("Validation_RMSE")
        .reset_index(drop=True)
    )

    return best_model, results


# ==========================================================
# 3. MODEL SELECTION
# ==========================================================

def best_hyperparameters(train, val, run_evaluation=False):
    """Train with known parameters or perform Grid Search."""

    if run_evaluation:

        model, results = grid_search(train, val)

    else:

        model = MatrixFactorizationSGD().fit(train, val)
        results = None

    params = {
        "k": model.k,
        "reg": model.reg,
        "learning_rate": model.learning_rate,
        "best_epoch": model.best_epoch,
        "seed": model.seed
    }

    return params, model, results


# ==========================================================
# 4. MAIN
# ==========================================================

def main(run_evaluation=False):

    # Load and split dataset
    ratings = load_ratings()
    train, val, test = split_ratings(ratings, seed=SEED)

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test (reserved): {len(test)}"
    )

    # Train with default parameters or perform Grid Search
    params, model, results = best_hyperparameters(
        train, val, run_evaluation=run_evaluation
    )

    # Show Grid Search results only when requested
    if run_evaluation:
        print("\n=== GRID SEARCH RESULTS ===")
        print(results.round(4).to_string(index=False))

    # Selected model
    print("\n=== SELECTED MODEL ===")
    print(f"Rank:            {params['k']}")
    print(f"Regularization:  {params['reg']}")
    print(f"Learning Rate:   {params['learning_rate']}")
    print(f"Best Epoch:      {params['best_epoch']}")
    print(f"Validation RMSE: {model.best_val_rmse:.4f}")

    # --------------------------------------------------
    # Training and Validation RMSE across epochs
    # --------------------------------------------------

    history = pd.DataFrame(model.history)

    print("\n=== CONVERGENCE HISTORY ===")
    print(history.round(4).to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(history["Epoch"], history["Train_RMSE"], label="Training RMSE")
    plt.plot(history["Epoch"], history["Validation_RMSE"], label="Validation RMSE")
    plt.axvline(model.best_epoch, linestyle="--", label="Best Epoch")

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("SGD Matrix Factorization - Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Grid Search to optimize hyperparameters"
    )

    args = parser.parse_args()
    main(run_evaluation=args.evaluate)
