
"""M5 - MovieLens 100K: Final rating prediction evaluation on the test set."""

import numpy as np
import pandas as pd

from baselines import load_ratings, split_ratings, BaselineRecommender
from matrix_factorization import best_hyperparameters


# ==========================================================
# 1. RATING PREDICTION METRICS
# ==========================================================

def evaluate_sgd(model, data):
    """Compute the same test metrics used for the baselines."""

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
# 2. MAIN
# ==========================================================

def main():

    # Same dataset split used in M3-M4
    ratings = load_ratings()
    train, val, test = split_ratings(ratings, seed=42)

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

    # --------------------------------------------------
    # 1. Select best model using Train and Validation
    # --------------------------------------------------

    params, model, _ = best_hyperparameters(train, val)

    print("\n=== SELECTED HYPERPARAMETERS ===")
    print(params)

    # --------------------------------------------------
    # 2. Evaluate SGD on Test
    # --------------------------------------------------

    sgd_results = evaluate_sgd(model, test)

    # --------------------------------------------------
    # 3. Evaluate Baselines on the same Test set
    # --------------------------------------------------

    baselines = BaselineRecommender(
        lambda_reg=10,
        n_iters=15
    ).fit(train)

    baseline_results = baselines.evaluate(test)

    # --------------------------------------------------
    # 4. Compare Test Results
    # --------------------------------------------------

    results = pd.concat(
        [baseline_results, sgd_results],
        ignore_index=True
    )

    results = (
        results.sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n=== M5: RATING PREDICTION ON TEST SET ===")
    print(results.round(4).to_string(index=False))

    # --------------------------------------------------
    # 5. Example Test Predictions
    # --------------------------------------------------

    examples = test.sample(
        n=min(5, len(test)),
        random_state=42
    ).copy()

    examples["prediction"] = model.predict(examples)

    print("\n=== TEST PREDICTIONS (SGD) ===")

    print(
        examples[
            ["user_id", "movie_id", "rating", "prediction"]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
