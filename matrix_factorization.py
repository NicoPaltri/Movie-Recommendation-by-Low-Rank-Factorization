
"""M3-M4 - MovieLens 100K: Matrix Factorization with SGD and Grid Search."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from baselines import load_ratings, split_ratings


# ==========================================================
# 1. MATRIX FACTORIZATION WITH SGD (M3)
# ==========================================================

class MatrixFactorizationSGD:

    def __init__(self, k=10, learning_rate=0.01,
                 reg=0.05, epochs=20, seed=42):

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

        # SGD training
        for epoch in range(self.epochs):

            # Shuffle observed ratings each epoch
            for pos in rng.permutation(len(ratings)):
                u = user_idx[pos]
                i = movie_idx[pos]
                r = ratings[pos]

                # Keep previous factors for both updates
                p = self.U[u].copy()
                q = self.V[i].copy()

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

            train_rmse = self.rmse(train)
            val_rmse = self.rmse(val)

            self.history.append({
                "Epoch": epoch + 1,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": val_rmse
            })

            # Save the model at the best validation epoch
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch + 1
                best_params = (
                    self.U.copy(),
                    self.V.copy(),
                    self.user_bias.copy(),
                    self.movie_bias.copy()
                )

        # Restore parameters from the best epoch
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

        pred[known_users] += self.user_bias[users[known_users]]
        pred[known_movies] += self.movie_bias[movies[known_movies]]

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

def grid_search(train, val, ranks, regularizations, epochs=20):
    """Find the best rank, regularization and epoch using validation RMSE."""

    results = []
    best_model = None
    best_rmse = float("inf")

    for k in ranks:
        for reg in regularizations:
            print(f"\nTraining model: k={k}, reg={reg}")

            model = MatrixFactorizationSGD(
                k=k,
                learning_rate=0.01,
                reg=reg,
                epochs=epochs,
                seed=42
            ).fit(train, val)

            results.append({
                "Rank": k,
                "Regularization": reg,
                "Best_Epoch": model.best_epoch,
                "Validation_RMSE": model.best_val_rmse
            })

            print(
                f"Best epoch: {model.best_epoch} | "
                f"Validation RMSE: {model.best_val_rmse:.4f}"
            )

            if model.best_val_rmse < best_rmse:
                best_rmse = model.best_val_rmse
                best_model = model

    results = (
        pd.DataFrame(results)
        .sort_values("Validation_RMSE")
        .reset_index(drop=True)
    )

    return best_model, results


def best_hyperparameters(train, val, run_evaluation=False):
    """Use best known parameters, or run Grid Search on request."""

    if run_evaluation:
        model, results = grid_search(
            train,
            val,
            ranks=[5, 10, 20],
            regularizations=[0.01, 0.05, 0.1],
            epochs=20
        )
    else:
        model = MatrixFactorizationSGD(
            k=10,
            learning_rate=0.01,
            reg=0.05,
            epochs=20,
            seed=42
        ).fit(train, val)

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
# 3. MAIN (M3-M4)
# ==========================================================

def main(run_evaluation=False):

    ratings = load_ratings()
    train, val, test = split_ratings(ratings, seed=42)

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test (reserved): {len(test)}"
    )

    if run_evaluation:
        print("\n=== RUNNING GRID SEARCH ===")
    else:
        print("\nUsing best known parameters: k=10, reg=0.05, epochs=20")

    params, model, results = best_hyperparameters(
        train, val, run_evaluation=run_evaluation
    )

    if run_evaluation:
        print("\n=== GRID SEARCH RESULTS ===")
        print(results.round(4).to_string(index=False))

    print("\n=== SELECTED MODEL ===")
    print(f"Rank:           {params['k']}")
    print(f"Regularization: {params['reg']}")
    print(f"Best Epoch:     {params['best_epoch']}")
    print(f"Validation RMSE: {model.best_val_rmse:.4f}")

    # Detailed evaluation only when requested
    if not run_evaluation:
        return

    # Training and validation RMSE across epochs
    history = pd.DataFrame(model.history)

    print("\n=== CONVERGENCE HISTORY (BEST MODEL) ===")
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

    # Validation metrics
    y = val["rating"].to_numpy(dtype=float)
    pred = model.predict(val)
    error = y - pred

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(error ** 2) / denom if denom > 0 else np.nan
    accuracy = np.mean(np.rint(pred) == y)
    within_one = np.mean(np.abs(np.rint(pred) - y) <= 1)

    print("\n=== BEST MATRIX FACTORIZATION RESULTS (VALIDATION) ===")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Within_1: {within_one:.4%}")

    examples = val.sample(n=min(5, len(val)), random_state=42).copy()
    examples["prediction"] = model.predict(examples)

    print("\n=== EXAMPLE PREDICTIONS (BEST SGD) ===")
    print(
        examples[["user_id", "movie_id", "rating", "prediction"]]
        .round(2).to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    main(run_evaluation=args.evaluate)
