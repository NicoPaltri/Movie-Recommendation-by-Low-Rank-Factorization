
import pandas as pd
from sklearn.model_selection import train_test_split

# Load MovieLens 100K
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# First split: 90% development, 10% test
development, test = train_test_split(
    ratings,
    test_size=0.10,
    stratify=ratings["user_id"],
    random_state=42
)

# Second split: 80% training, 10% validation
train, val = train_test_split(
    development,
    test_size=1/9,
    stratify=development["user_id"],
    random_state=42
)

# Ensure that every movie is represented in training
missing_movies = (
    set(ratings["movie_id"]) - set(train["movie_id"])
)

for movie in sorted(missing_movies):
    # Move one observation of the movie into training
    if (val["movie_id"] == movie).any():
        chosen = val[val["movie_id"] == movie].sample(
            n=1, random_state=42
        )
        val = val.drop(chosen.index)
    else:
        chosen = test[test["movie_id"] == movie].sample(
            n=1, random_state=42
        )
        test = test.drop(chosen.index)

    train = pd.concat([train, chosen])

# Verify warm-start coverage
for subset in [val, test]:
    assert set(subset["user_id"]) <= set(train["user_id"])
    assert set(subset["movie_id"]) <= set(train["movie_id"])

# Verify that observations are disjoint
assert len(set(train.index) & set(val.index)) == 0
assert len(set(train.index) & set(test.index)) == 0
assert len(set(val.index) & set(test.index)) == 0

print("Training:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
