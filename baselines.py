from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. DATASET
# ==========================================================

DATA_PATH = Path(__file__).resolve().parent / "ml-100k" / "u.data"

COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]

METHODS = (
    "global_mean",
    "user_mean",
    "item_mean",
    "bias_sequential",
    "user_item_bias",
)


def load_ratings():
    """Carica MovieLens 100K dalla cartella locale ml-100k."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"File non trovato: {DATA_PATH}\n"
            "Verifica che ml-100k/u.data sia accanto a baselines.py."
        )

    return pd.read_csv(DATA_PATH, sep="\t", names=COLUMNS)


# ==========================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

def split_ratings(ratings, seed=42):
    """Split 80/10/10, stratificato per utente."""

    train, remaining = train_test_split(
        ratings,
        test_size=0.20,
        stratify=ratings["user_id"],
        random_state=seed
    )

    val, test = train_test_split(
        remaining,
        test_size=0.50,
        stratify=remaining["user_id"],
        random_state=seed
    )

    return train, val, test


# ==========================================================
# 3. BASELINE MODELS
# ==========================================================

class BaselineRecommender:
    """Quattro baseline principali + bias sequenziale.

    bias_sequential:
        Stima prima i bias dei film e poi quelli degli utenti.

    user_item_bias:
        Stima alternata con regolarizzazione L2.
    """

    def __init__(self, lambda_reg=10.0, n_iters=15):
        self.lambda_reg = lambda_reg
        self.n_iters = n_iters

    def fit(self, train):

        # Global mean
        self.global_mean = train["rating"].mean()

        # User and item means
        self.user_means = train.groupby("user_id")["rating"].mean()
        self.item_means = train.groupby("movie_id")["rating"].mean()

        # --------------------------------------------------
        # Sequential bias (approccio della compagna)
        # --------------------------------------------------

        self.seq_item_bias = self.item_means - self.global_mean

        residual = (
            train["rating"]
            - self.global_mean
            - train["movie_id"].map(self.seq_item_bias)
        )

        self.seq_user_bias = residual.groupby(
            train["user_id"]
        ).mean()

        # --------------------------------------------------
        # Regularized alternating bias (primo approccio)
        # --------------------------------------------------

        user_counts = train.groupby("user_id").size()
        item_counts = train.groupby("movie_id").size()

        self.user_bias = pd.Series(
            0.0, index=user_counts.index
        )

        self.item_bias = pd.Series(
            0.0, index=item_counts.index
        )

        for _ in range(self.n_iters):

            # Update user bias
            user_residual = (
                train["rating"]
                - self.global_mean
                - train["movie_id"].map(self.item_bias)
            )

            self.user_bias = (
                user_residual.groupby(train["user_id"]).sum()
                / (user_counts + self.lambda_reg)
            )

            # Update item bias
            item_residual = (
                train["rating"]
                - self.global_mean
                - train["user_id"].map(self.user_bias)
            )

            self.item_bias = (
                item_residual.groupby(train["movie_id"]).sum()
                / (item_counts + self.lambda_reg)
            )

        return self

    def predict(self, data, method="user_item_bias"):
        """Predice i rating, applicando i fallback cold-start."""

        users = data["user_id"]
        items = data["movie_id"]
        mu = self.global_mean

        if method == "global_mean":
            pred = np.full(len(data), mu)

        elif method == "user_mean":
            pred = (
                users.map(self.user_means)
                .fillna(mu)
                .to_numpy()
            )

        elif method == "item_mean":
            pred = (
                items.map(self.item_means)
                .fillna(mu)
                .to_numpy()
            )

        elif method == "bias_sequential":
            pred = (
                mu
                + users.map(self.seq_user_bias).fillna(0).to_numpy()
                + items.map(self.seq_item_bias).fillna(0).to_numpy()
            )

        elif method == "user_item_bias":
            pred = (
                mu
                + users.map(self.user_bias).fillna(0).to_numpy()
                + items.map(self.item_bias).fillna(0).to_numpy()
            )

        else:
            raise ValueError(f"Metodo sconosciuto: {method}")

        # Ratings must be between 1 and 5
        return np.clip(pred, 1, 5)

    def evaluate(self, data):
        """Calcola le metriche per tutte le baseline."""

        y = data["rating"].to_numpy(dtype=float)
        results = []

        for method in METHODS:

            pred = self.predict(data, method)
            residual = y - pred

            denom = np.sum((y - y.mean()) ** 2)

            results.append({
                "Model": method,
                "RMSE": np.sqrt(np.mean(residual ** 2)),
                "MAE": np.mean(np.abs(residual)),
                "R2": (
                    1 - np.sum(residual ** 2) / denom
                    if denom > 0 else np.nan
                ),
                "Accuracy": np.mean(np.rint(pred) == y),
                "Within_1": np.mean(
                    np.abs(np.rint(pred) - y) <= 1
                ),
            })

        return (
            pd.DataFrame(results)
            .sort_values("RMSE")
            .reset_index(drop=True)
        )


# ==========================================================
# 4. MATRIX SPARSITY
# ==========================================================

def print_sparsity(ratings, train, val, test):
    """Calcola sparsity e density della matrice user-item."""

    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()

    total = n_users * n_movies

    print(f"\n=== SPARSITY: {n_users} users x {n_movies} movies ===")

    for name, subset in (
        ("Full dataset", ratings),
        ("Training", train),
        ("Validation", val),
        ("Test", test),
    ):

        observed = subset[
            ["user_id", "movie_id"]
        ].drop_duplicates().shape[0]

        density = observed / total
        sparsity = 1 - density

        print(
            f"{name:<13} | "
            f"Observed: {observed:>6} | "
            f"Density: {density:>6.2%} | "
            f"Sparsity: {sparsity:>6.2%}"
        )


# ==========================================================
# 5. COLD START ANALYSIS
# ==========================================================

def cold_start_types(data, train):
    """Classifica ogni osservazione come warm o cold-start."""

    known_user = data["user_id"].isin(train["user_id"])
    known_item = data["movie_id"].isin(train["movie_id"])

    return np.select(
        [
            known_user & known_item,
            ~known_user & known_item,
            known_user & ~known_item
        ],
        [
            "warm",
            "new_user",
            "new_item"
        ],
        default="both_new"
    )


def print_cold_start(data, train, name):
    """Stampa quanti casi warm e cold-start sono presenti."""

    counts = pd.Series(
        cold_start_types(data, train)
    ).value_counts()

    print(f"\n=== {name}: WARM / COLD START ===")

    for case in (
        "warm",
        "new_user",
        "new_item",
        "both_new"
    ):
        print(f"{case:<10}: {counts.get(case, 0):>6}")


# ==========================================================
# 6. AVERAGE RESULTS
# ==========================================================

def average_results(ratings, seeds=(42, 43, 44, 45, 46)):
    """Media e deviazione standard su split differenti."""

    runs = []

    for seed in seeds:

        train, val, _ = split_ratings(ratings, seed=seed)

        model = BaselineRecommender()
        model.fit(train)

        result = model.evaluate(val)
        result["Seed"] = seed

        runs.append(result)

    runs = pd.concat(runs, ignore_index=True)

    metrics = [
        "RMSE",
        "MAE",
        "R2",
        "Accuracy",
        "Within_1"
    ]

    summary = (
        runs.groupby("Model")[metrics]
        .agg(["mean", "std"])
    )

    return runs, summary


# ==========================================================
# 7. MAIN
# ==========================================================


def main():
    # 1. Load dataset
    ratings = load_ratings()

    # 2. Split dataset (80/10/10)
    train, val, test = split_ratings(ratings)

    print("\n=== DATASET SPLIT ===")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    # 3. Matrix sparsity
    print_sparsity(ratings, train, val, test)

    # 4. Cold start analysis
    print_cold_start(val, train, "Validation")
    print_cold_start(test, train, "Test")

    # 5. Train baselines
    model = BaselineRecommender(
        lambda_reg=10,
        n_iters=15
    )
    model.fit(train)

    # 6. Evaluate on validation set
    results = model.evaluate(val)

    print("\n=== BASELINE COMPARISON ===")
    print(results.round(4).to_string(index=False))

    # 7. Example predictions using the best baseline
    best_method = results.iloc[0]["Model"]

    examples = val.sample(n=5, random_state=42).copy()

    examples["prediction"] = model.predict(
        examples,
        method=best_method
    )

    print("\n=== EXAMPLE PREDICTIONS ===")
    print(
        examples[
            ["user_id", "movie_id", "rating", "prediction"]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
