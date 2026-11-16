
"""M5 - Rating prediction, Top-N ranking and SGD scalability."""

import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import load_ratings, split_ratings, BaselineRecommender
from matrix_factorization import MatrixFactorizationSGD, best_hyperparameters


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42
TOP_N = 10
RELEVANT_RATING = 4

BENCHMARK_RANKS = [5, 10, 20, 30]
BENCHMARK_FRACTIONS = [0.25, 0.50, 1.0]
BENCHMARK_EPOCHS = 5


# ==========================================================
# 1. RATING PREDICTION METRICS
# ==========================================================

def evaluate_sgd(model, data):
    """Compute the same metrics used for the M2 baselines."""

    y = data["rating"].to_numpy(dtype=float)
    pred = model.predict(data)
    error = y - pred

    denom = np.sum((y - y.mean()) ** 2)

    return pd.DataFrame([{
        "Model": "matrix_factorization_sgd",
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MAE": np.mean(np.abs(error)),
        "R2": (
            1 - np.sum(error ** 2) / denom
            if denom > 0 else np.nan
        ),
        "Accuracy": np.mean(np.rint(pred) == y),
        "Within_1": np.mean(
            np.abs(np.rint(pred) - y) <= 1
        )
    }])


# ==========================================================
# 2. TOP-N RANKING
# ==========================================================

def evaluate_top_n(predict_fn, train, val, test, n=TOP_N):
    """Evaluate Recall@N and HitRate@N on the test set."""

    # All movies appearing in the training set
    catalog = train["movie_id"].unique()
    known_users = set(train["user_id"].unique())

    # Movies already rated by each user
    seen = (
        pd.concat([
            train[["user_id", "movie_id"]],
            val[["user_id", "movie_id"]]
        ])
        .groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    # Relevant test movies: rating >= 4
    positives = test[test["rating"] >= RELEVANT_RATING]

    # Keep only users and movies known in training
    eligible = positives[
        positives["user_id"].isin(known_users)
        & positives["movie_id"].isin(catalog)
    ]

    relevant_by_user = (
        eligible.groupby("user_id")["movie_id"].agg(set)
    )

    recalls = []
    hits = []

    for user_id, relevant_movies in relevant_by_user.items():

        # Movies not already rated by this user
        candidates = catalog[
            ~np.isin(catalog, list(seen.get(user_id, set())))
        ]

        if len(candidates) == 0:
            continue

        # Predict ratings for all candidate movies
        pairs = pd.DataFrame({
            "user_id": user_id,
            "movie_id": candidates
        })

        scores = predict_fn(pairs)

        # Select the N highest-rated predictions
        recommended = set(
            candidates[np.argsort(-scores, kind="stable")[:n]]
        )

        # Count relevant recommendations
        n_hits = len(recommended & relevant_movies)

        recalls.append(n_hits / len(relevant_movies))
        hits.append(n_hits > 0)

    return {
        f"Recall@{n}": np.mean(recalls) if recalls else np.nan,
        f"HitRate@{n}": np.mean(hits) if hits else np.nan,
        "Users": len(recalls),
        "Eligible_Positives": len(eligible),
        "Excluded_Positives": len(positives) - len(eligible)
    }


# ==========================================================
# 3. RUNTIME AND MEMORY BENCHMARK
# ==========================================================

def benchmark_scaling(train, val, params):
    """Compare training time and memory across ranks and data sizes."""

    rows = []

    for fraction in BENCHMARK_FRACTIONS:

        subset = (
            train if fraction == 1.0
            else train.sample(frac=fraction, random_state=SEED)
        )

        for k in BENCHMARK_RANKS:

            model = MatrixFactorizationSGD(
                k=k,
                reg=params["reg"],
                learning_rate=params["learning_rate"],
                epochs=BENCHMARK_EPOCHS,
                seed=SEED
            )

            # Training time
            start = perf_counter()
            model.fit(subset, val)
            elapsed = perf_counter() - start

            # Memory of model parameters
            model_bytes = sum(
                array.nbytes for array in (
                    model.U,
                    model.V,
                    model.user_bias,
                    model.movie_bias
                )
            )

            # Memory of training DataFrame
            data_bytes = subset.memory_usage(deep=True).sum()

            rows.append({
                "Train_Fraction": fraction,
                "Train_Ratings": len(subset),
                "Rank": k,
                "Users": len(model.user_ids),
                "Movies": len(model.movie_ids),
                "Time_s": elapsed,
                "Model_MB": model_bytes / 1024**2,
                "TrainData_MB": data_bytes / 1024**2,
                "Tracked_MB": (model_bytes + data_bytes) / 1024**2
            })

            print(
                f"Benchmark: {fraction:.0%} train, "
                f"k={k}, time={elapsed:.2f}s"
            )

    return pd.DataFrame(rows)


def plot_benchmark(results):
    """Plot runtime and tracked memory against rank."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for fraction, group in results.groupby("Train_Fraction"):

        group = group.sort_values("Rank")
        label = f"{fraction:.0%} training ratings"

        axes[0].plot(
            group["Rank"],
            group["Time_s"],
            marker="o",
            label=label
        )

        axes[1].plot(
            group["Rank"],
            group["Tracked_MB"],
            marker="o",
            label=label
        )

    axes[0].set_title("SGD training time")
    axes[0].set_ylabel("Seconds")

    axes[1].set_title("Tracked memory (model + training data)")
    axes[1].set_ylabel("MiB")

    for ax in axes:
        ax.set_xlabel("Rank k")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# 4. MAIN
# ==========================================================

def main(run_benchmark=False):

    # Load and split dataset
    ratings = load_ratings()
    train, val, test = split_ratings(ratings, seed=SEED)

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

    # --------------------------------------------------
    # 1. Train selected SGD model and M2 baselines
    # --------------------------------------------------

    params, model, _ = best_hyperparameters(train, val)

    print("\n=== SELECTED HYPERPARAMETERS ===")
    print(params)

    baselines = BaselineRecommender(
        lambda_reg=10,
        n_iters=15
    ).fit(train)

    # --------------------------------------------------
    # 2. Rating prediction on Test
    # --------------------------------------------------

    results = pd.concat([
        baselines.evaluate(test),
        evaluate_sgd(model, test)
    ], ignore_index=True)

    results = (
        results.sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n=== M5: RATING PREDICTION (TEST) ===")
    print(results.round(4).to_string(index=False))

    # --------------------------------------------------
    # 3. Top-N ranking on Test
    # --------------------------------------------------

    ranking = pd.DataFrame([
        {
            "Model": "matrix_factorization_sgd",
            **evaluate_top_n(
                model.predict,
                train, val, test
            )
        },
        {
            "Model": "user_item_bias",
            **evaluate_top_n(
                lambda pairs: baselines.predict(
                    pairs, method="user_item_bias"
                ),
                train, val, test
            )
        }
    ])

    print(
        f"\n=== M5: TOP-{TOP_N} RANKING "
        f"(TEST; POSITIVE >= {RELEVANT_RATING}) ==="
    )
    print(ranking.round(4).to_string(index=False))

    # --------------------------------------------------
    # 4. Reconstruction Error vs Top-N Ranking
    # --------------------------------------------------

    # Training RMSE (observed ratings)
    train_rmse = pd.concat([
        baselines.evaluate(train)[["Model", "RMSE"]],
        evaluate_sgd(model, train)[["Model", "RMSE"]]
    ], ignore_index=True).rename(
        columns={"RMSE": "Train_RMSE"}
    )

    # Validation RMSE
    val_rmse = pd.concat([
        baselines.evaluate(val)[["Model", "RMSE"]],
        evaluate_sgd(model, val)[["Model", "RMSE"]]
    ], ignore_index=True).rename(
        columns={"RMSE": "Validation_RMSE"}
    )

    # Combine training, validation, test and ranking
    comparison = (
        ranking[["Model", f"Recall@{TOP_N}", f"HitRate@{TOP_N}"]]
        .merge(train_rmse, on="Model")
        .merge(val_rmse, on="Model")
        .merge(
            results[["Model", "RMSE"]].rename(
                columns={"RMSE": "Test_RMSE"}
            ),
            on="Model"
        )
    )

    comparison = comparison[[
        "Model",
        "Train_RMSE",
        "Validation_RMSE",
        "Test_RMSE",
        f"Recall@{TOP_N}",
        f"HitRate@{TOP_N}"
    ]]

    print("\n=== M5: RECONSTRUCTION ERROR VS TOP-N RANKING ===")
    print(comparison.round(4).to_string(index=False))

    # --------------------------------------------------
    # 5. Example Test predictions
    # --------------------------------------------------

    examples = test.sample(
        n=min(5, len(test)),
        random_state=SEED
    ).copy()

    examples["prediction"] = model.predict(examples)

    print("\n=== EXAMPLE TEST PREDICTIONS (SGD) ===")
    print(
        examples[
            ["user_id", "movie_id", "rating", "prediction"]
        ].round(2).to_string(index=False)
    )

    # --------------------------------------------------
    # 6. Runtime and memory benchmark
    # --------------------------------------------------

    if run_benchmark:

        print(
            f"\n=== M5: SCALING BENCHMARK "
            f"({BENCHMARK_EPOCHS} EPOCHS) ==="
        )

        performance = benchmark_scaling(train, val, params)

        print(performance.round(4).to_string(index=False))

        plot_benchmark(performance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare SGD runtime and memory across ranks and data sizes"
    )

    args = parser.parse_args()
    main(run_benchmark=args.benchmark)
