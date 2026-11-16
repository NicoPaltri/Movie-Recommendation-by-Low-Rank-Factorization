
"""M6 - Popularity bias, cold-start behavior and latent factor analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import load_ratings, split_ratings, BaselineRecommender
from matrix_factorization import best_hyperparameters


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42
TOP_N = 10
POPULAR_FRACTION = 0.10

FACTORS_TO_INSPECT = [0, 1]
MOVIES_PER_SIDE = 5

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


# ==========================================================
# 1. POPULARITY BIAS
# ==========================================================

def analyze_popularity(model, baselines, train, val):
    """Compare popularity and diversity of Top-N recommendations."""

    # Popularity = number of observed training ratings per movie
    popularity = train["movie_id"].value_counts()
    catalog = model.movie_ids.to_numpy()

    n_popular = max(1, int(np.ceil(len(catalog) * POPULAR_FRACTION)))
    popular_movies = set(popularity.nlargest(n_popular).index)

    # Movies already rated in training and validation
    seen = (
        pd.concat([
            train[["user_id", "movie_id"]],
            val[["user_id", "movie_id"]]
        ])
        .groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    predictors = {
        "matrix_factorization_sgd": model.predict,
        "user_item_bias": lambda pairs: baselines.predict(
            pairs, method="user_item_bias"
        )
    }

    rows = []

    for name, predict_fn in predictors.items():

        recommended_movies = []

        for user_id in model.user_ids:

            # All movies not already rated by this user
            candidates = catalog[
                ~np.isin(catalog, list(seen.get(user_id, set())))
            ]

            if len(candidates) == 0:
                continue

            # Predict ratings for the candidate movies
            pairs = pd.DataFrame({
                "user_id": user_id,
                "movie_id": candidates
            })

            scores = predict_fn(pairs)

            # Select the Top-N recommendations
            top_movies = candidates[
                np.argsort(-scores, kind="stable")[:TOP_N]
            ]

            recommended_movies.extend(top_movies.tolist())

        recommended_movies = np.asarray(recommended_movies)
        movie_popularity = popularity.reindex(
            recommended_movies
        ).to_numpy()

        rows.append({
            "Model": name,
            "Mean_Popularity": np.mean(movie_popularity),
            "Median_Popularity": np.median(movie_popularity),
            "Top_10pct_Share": np.mean(
                np.isin(recommended_movies, list(popular_movies))
            ),
            "Unique_Movies": len(np.unique(recommended_movies)),
            "Catalog_Coverage": (
                len(np.unique(recommended_movies)) / len(catalog)
            )
        })

    results = pd.DataFrame(rows)

    print("\n=== M6: POPULARITY BIAS (TOP-10 RECOMMENDATIONS) ===")
    print(f"Training catalog: {len(catalog)} movies")
    print(f"Top popularity group: {n_popular} most-rated movies")
    print(results.round(4).to_string(index=False))

    # Graphical comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(
        results["Model"],
        results["Top_10pct_Share"] * 100
    )
    axes[0].set_title("Recommendations from top 10% popular movies")
    axes[0].set_ylabel("Share of recommendations (%)")

    axes[1].bar(
        results["Model"],
        results["Catalog_Coverage"] * 100
    )
    axes[1].set_title("Catalog coverage")
    axes[1].set_ylabel("Movies recommended at least once (%)")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==========================================================
# 2. COLD-START BEHAVIOR
# ==========================================================

def analyze_cold_start(model, baselines, train, test):
    """Show cold-start fallback predictions and test-set coverage."""

    known_user = int(train.iloc[0]["user_id"])
    known_movie = int(train.iloc[0]["movie_id"])

    new_user = int(train["user_id"].max()) + 1
    new_movie = int(train["movie_id"].max()) + 1

    # Artificial examples to verify fallback behavior
    examples = pd.DataFrame({
        "Case": [
            "Known / Known",
            "New / Known",
            "Known / New",
            "New / New"
        ],
        "user_id": [
            known_user, new_user, known_user, new_user
        ],
        "movie_id": [
            known_movie, known_movie, new_movie, new_movie
        ]
    })

    examples["SGD_Prediction"] = model.predict(examples)

    examples["Bias_Prediction"] = baselines.predict(
        examples, method="user_item_bias"
    )

    print("\n=== M6: COLD-START FALLBACK EXAMPLES ===")
    print(examples.round(4).to_string(index=False))

    # Analyze real cold-start cases in the test set
    known_users = test["user_id"].isin(
        model.user_ids
    ).to_numpy()

    known_movies = test["movie_id"].isin(
        model.movie_ids
    ).to_numpy()

    groups = {
        "Known / Known": known_users & known_movies,
        "New / Known": ~known_users & known_movies,
        "Known / New": known_users & ~known_movies,
        "New / New": ~known_users & ~known_movies
    }

    rows = []

    for name, mask in groups.items():

        subset = test.loc[mask]

        if len(subset):

            actual = subset["rating"].to_numpy(dtype=float)

            sgd_rmse = np.sqrt(
                np.mean((actual - model.predict(subset)) ** 2)
            )

            bias_pred = baselines.predict(
                subset, method="user_item_bias"
            )

            bias_rmse = np.sqrt(
                np.mean((actual - bias_pred) ** 2)
            )

        else:
            sgd_rmse = np.nan
            bias_rmse = np.nan

        rows.append({
            "Case": name,
            "Test_Ratings": len(subset),
            "SGD_RMSE": sgd_rmse,
            "Bias_RMSE": bias_rmse
        })

    print("\n=== M6: COLD-START CASES IN TEST ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("Note: empty groups have no measurable test RMSE.")


# ==========================================================
# 3. LATENT FACTOR INTERPRETATION
# ==========================================================

def inspect_latent_factors(model):
    """Explore movies with extreme values along selected latent axes."""

    item_path = (
        Path(__file__).resolve().parent
        / "ml-100k"
        / "u.item"
    )

    if not item_path.exists():
        print(f"\nMovie metadata not found: {item_path}")
        print("Skipping latent factor interpretation.")
        return

    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url"
    ] + GENRES

    # Load MovieLens titles and genre metadata
    movies = pd.read_csv(
        item_path,
        sep="|",
        header=None,
        names=columns,
        encoding="latin-1"
    ).set_index("movie_id")

    print("\n=== M6: LATENT FACTOR EXPLORATION ===")

    for factor in FACTORS_TO_INSPECT:

        if factor >= model.k:
            continue

        values = model.V[:, factor]
        order = np.argsort(values)

        print(f"\n--- Latent dimension {factor} ---")

        # Examine movies at both extremes of this dimension
        for label, positions in (
            ("Highest factor values", order[-MOVIES_PER_SIDE:][::-1]),
            ("Lowest factor values", order[:MOVIES_PER_SIDE])
        ):

            rows = []

            for pos in positions:

                movie_id = int(model.movie_ids[pos])
                movie = movies.loc[movie_id]

                genres = ", ".join(
                    genre for genre in GENRES
                    if movie[genre] == 1
                )

                rows.append({
                    "Movie": movie["title"],
                    "Genres": genres,
                    "Factor_Value": values[pos]
                })

            print(f"\n{label}:")
            print(
                pd.DataFrame(rows)
                .round(3)
                .to_string(index=False)
            )

    print("\nThese dimensions are not automatically genres or preferences.")
    print("Factor signs/axes are not unique; any interpretation is exploratory.")


# ==========================================================
# 4. MAIN
# ==========================================================

def main():

    # Load and split dataset
    ratings = load_ratings()
    train, val, test = split_ratings(ratings, seed=SEED)

    # Reuse the model selected on validation in M4
    params, model, _ = best_hyperparameters(train, val)

    # Train M2 baseline for comparison
    baselines = BaselineRecommender(
        lambda_reg=10,
        n_iters=15
    ).fit(train)

    print("=== M6: SELECTED MODEL ===")
    print(params)

    # 1. Popularity bias
    analyze_popularity(model, baselines, train, val)

    # 2. Cold-start behavior
    analyze_cold_start(model, baselines, train, test)

    # 3. Latent factor interpretation
    inspect_latent_factors(model)


if __name__ == "__main__":
    main()
