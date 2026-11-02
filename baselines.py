
"""Simple rating-prediction baselines for MovieLens 100K.

Models:
    global_mean:    r_hat(u,i) = mu
    user_mean:      r_hat(u,i) = mean rating by user u
    item_mean:      r_hat(u,i) = mean rating for movie i
    user_item_bias: r_hat(u,i) = mu + b_u + b_i

Fit the models ONLY on the training set; never on validation/test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaselineRecommender:
    def __init__(self, lambda_reg=10.0, n_iters=15, clip=True):
        """Parameters
        ----------
        lambda_reg : float
            L2 regularization strength for user/item biases.
        n_iters : int
            Number of alternating bias-estimation iterations.
        clip : bool
            If True, clip predictions to the MovieLens rating scale [1, 5].
        """
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1")

        self.lambda_reg = lambda_reg
        self.n_iters = n_iters
        self.clip = clip
        self.fitted = False

    def fit(self, train):
        """Fit all four baselines using a DataFrame of observed ratings."""
        columns = {"user_id", "movie_id", "rating"}

        if not columns.issubset(train.columns):
            raise ValueError(f"train must contain columns: {sorted(columns)}")
        if train.empty:
            raise ValueError("train must not be empty")

        train = train[["user_id", "movie_id", "rating"]].copy()

        # 1. Global mean
        self.global_mean = train["rating"].mean()

        # 2. User mean and 3. Item mean
        self.user_means = train.groupby("user_id")["rating"].mean()
        self.item_means = train.groupby("movie_id")["rating"].mean()

        # 4. Regularized user + item bias:
        # minimize sum_(u,i) (r_ui - mu - b_u - b_i)^2
        #          + lambda_reg * (sum_u b_u^2 + sum_i b_i^2)

        user_counts = train.groupby("user_id").size()
        item_counts = train.groupby("movie_id").size()

        self.user_bias = pd.Series(0.0, index=user_counts.index)
        self.item_bias = pd.Series(0.0, index=item_counts.index)

        for _ in range(self.n_iters):
            # Update user biases, keeping item biases fixed
            item_effect = train["movie_id"].map(self.item_bias)
            user_residual = train["rating"] - self.global_mean - item_effect

            self.user_bias = (
                user_residual.groupby(train["user_id"]).sum()
                / (user_counts + self.lambda_reg)
            )

            # Update item biases, keeping user biases fixed
            user_effect = train["user_id"].map(self.user_bias)
            item_residual = train["rating"] - self.global_mean - user_effect

            self.item_bias = (
                item_residual.groupby(train["movie_id"]).sum()
                / (item_counts + self.lambda_reg)
            )

        self.fitted = True
        return self

    def predict(self, data, method="global_mean"):
        """Predict one rating per row."""
        if not self.fitted:
            raise RuntimeError("Call fit(train) before predict().")

        users = data["user_id"]
        items = data["movie_id"]

        if method == "global_mean":
            predictions = np.full(len(data), self.global_mean)

        elif method == "user_mean":
            predictions = (
                users.map(self.user_means)
                .fillna(self.global_mean)
                .to_numpy()
            )

        elif method == "item_mean":
            predictions = (
                items.map(self.item_means)
                .fillna(self.global_mean)
                .to_numpy()
            )

        elif method == "user_item_bias":
            user_effect = users.map(self.user_bias).fillna(0.0).to_numpy()
            item_effect = items.map(self.item_bias).fillna(0.0).to_numpy()

            predictions = self.global_mean + user_effect + item_effect

        else:
            raise ValueError(
                "Unknown method. Choose among: global_mean, user_mean, "
                "item_mean, user_item_bias"
            )

        if self.clip:
            predictions = np.clip(predictions, 1.0, 5.0)

        return predictions

    def evaluate(self, data):
        """Compute RMSE and MAE for each baseline."""
        if "rating" not in data.columns:
            raise ValueError("data must contain the rating column")
        if data.empty:
            raise ValueError("data must not be empty")

        y_true = data["rating"].to_numpy(dtype=float)
        results = []

        for method in (
            "global_mean",
            "user_mean",
            "item_mean",
            "user_item_bias"
        ):
            y_pred = self.predict(data, method=method)

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))

            results.append({
                "model": method,
                "RMSE": rmse,
                "MAE": mae
            })

        return (
            pd.DataFrame(results)
            .sort_values("RMSE")
            .reset_index(drop=True)
        )


def print_matrix_sparsity(ratings, train, val, test):
    """Compute matrix density and sparsity for each dataset split."""

    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()

    # Total number of possible user-movie combinations
    n_possible = n_users * n_movies

    print(f"\n=== MATRIX SPARSITY ({n_users} users x {n_movies} movies) ===")
    print(f"Possible user-movie pairs: {n_possible:,}")

    for name, subset in (
        ("Full dataset", ratings),
        ("Training", train),
        ("Validation", val),
        ("Test", test),
    ):
        # Count observed user-movie pairs
        n_observed = len(
            subset[["user_id", "movie_id"]].drop_duplicates()
        )

        # Density = observed entries / total possible entries
        density = n_observed / n_possible

        # Sparsity = missing entries / total possible entries
        sparsity = 1.0 - density

        print(
            f"{name:<13} | Observed: {n_observed:>7,} "
            f"| Density: {density:>7.2%} "
            f"| Sparsity: {sparsity:>7.2%}"
        )


def test_baselines(train, val):
    """Train and compare all four baselines on the validation set."""

    model = BaselineRecommender(lambda_reg=10, n_iters=15)
    model.fit(train)

    methods = [
        "global_mean",
        "user_mean",
        "item_mean",
        "user_item_bias"
    ]

    y_true = val["rating"].to_numpy()
    results = []

    for method in methods:
        # Predict ratings
        y_pred = model.predict(val, method=method)
        y_rounded = np.rint(y_pred)

        # Regression metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
            (y_true - y_true.mean()) ** 2
        )

        # Classification-like metrics
        accuracy = np.mean(y_rounded == y_true)
        within_one = np.mean(np.abs(y_rounded - y_true) <= 1)

        results.append({
            "Model": method,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Accuracy": accuracy,
            "Within_1": within_one
        })

    results = pd.DataFrame(results).sort_values("RMSE")

    print("\n=== BASELINE COMPARISON ===")
    print(results.round(4).to_string(index=False))

    # Select the best baseline
    best_method = results.iloc[0]["Model"]

    # Generate example predictions
    examples = val.sample(
        n=min(5, len(val)),
        random_state=42
    ).copy()

    examples["predicted_rating"] = model.predict(
        examples,
        method=best_method
    ).round(2)

    print(f"\n=== PREDICTIONS ({best_method}) ===")
    print(
        examples[
            ["user_id", "movie_id", "rating", "predicted_rating"]
        ].to_string(index=False)
    )

    return model, results, examples


if __name__ == "__main__":

    # Load MovieLens dataset
    dataset_path = Path(__file__).resolve().parent / "ml-100k" / "u.data"

    ratings = pd.read_csv(
        dataset_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Split: 80% train, 10% validation, 10% test
    development, test = train_test_split(
        ratings,
        test_size=0.10,
        stratify=ratings["user_id"],
        random_state=42
    )

    train, val = train_test_split(
        development,
        test_size=1/9,
        stratify=development["user_id"],
        random_state=42
    )

    # Ensure all movies are represented in training
    missing_movies = (
        set(ratings["movie_id"]) - set(train["movie_id"])
    )

    for movie in sorted(missing_movies):

        if (val["movie_id"] == movie).any():
            chosen = val[val["movie_id"] == movie].sample(
                n=1,
                random_state=42
            )
            val = val.drop(chosen.index)

        else:
            chosen = test[test["movie_id"] == movie].sample(
                n=1,
                random_state=42
            )
            test = test.drop(chosen.index)

        train = pd.concat([train, chosen])

    # Dataset summary
    print("\n=== DATASET SPLIT ===")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    # Compute matrix sparsity
    print_matrix_sparsity(ratings, train, val, test)

    # Run baseline evaluation
    model, results, examples = test_baselines(train, val)
