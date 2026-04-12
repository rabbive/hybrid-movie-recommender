# ============================================
# HYBRID MOVIE RECOMMENDATION SYSTEM
# ============================================

import ast
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split

from recommender_core import hybrid_recommend


def convert(text):
    try:
        return [i["name"].replace(" ", "") for i in ast.literal_eval(text)]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i.get("job") == "Director":
                return i["name"].replace(" ", "")
    except (ValueError, SyntaxError, TypeError, KeyError):
        return ""
    return ""


def train_all() -> None:
    # Load TMDB datasets
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
        ]
    ]

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert)
    movies["director"] = movies["crew"].apply(get_director)
    movies["overview"] = movies["overview"].fillna("")
    movies["director"] = movies["director"].fillna("")
    movies = movies.drop(columns=["crew"])
    movies["cast"] = movies["cast"].apply(lambda x: x[:3])
    movies["genres"] = movies["genres"].apply(lambda x: " ".join(x))
    movies["keywords"] = movies["keywords"].apply(lambda x: " ".join(x))
    movies["cast"] = movies["cast"].apply(lambda x: " ".join(x))
    movies["features"] = (
        movies["genres"]
        + " "
        + movies["keywords"]
        + " "
        + movies["cast"]
        + " "
        + movies["director"]
        + " "
        + movies["overview"]
    )
    movies["features"] = movies["features"].str.lower()

    ratings = pd.read_csv("ratings_small.csv")[["userId", "movieId", "rating"]]

    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    user_movie_matrix.fillna(0, inplace=True)

    links = pd.read_csv("links_small.csv")[["movieId", "tmdbId"]].dropna()
    links["tmdbId"] = links["tmdbId"].astype(int)
    movies = movies.merge(links, left_on="movie_id", right_on="tmdbId")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movies = movies.reset_index(drop=True)
    movies["title"] = movies["title"].str.lower()
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    cv_results = cross_validate(
        SVD(), data, measures=["RMSE"], cv=3, verbose=False
    )
    cv_metrics = {
        "test_rmse_mean": float(cv_results["test_rmse"].mean()),
        "test_rmse_std": float(cv_results["test_rmse"].std()),
        "n_splits": 3,
    }
    with open("cv_metrics.pkl", "wb") as f:
        pickle.dump(cv_metrics, f)

    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    with open("rmse.pkl", "wb") as f:
        pickle.dump(rmse, f)

    n_users = int(ratings["userId"].nunique())
    n_movies_ml = int(ratings["movieId"].nunique())
    n_ratings = int(len(ratings))
    cells = n_users * n_movies_ml
    sparsity = float(1.0 - n_ratings / cells) if cells else 0.0

    dataset_stats = {
        "n_users": n_users,
        "n_movies_movielens": n_movies_ml,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "n_movies_after_join": int(len(movies)),
        "train_test_split": "80% train / 20% test (single split for saved model)",
        "cross_validate_splits": 3,
        "rating_scale": (0.5, 5.0),
        "svd_n_factors": int(model.n_factors),
        "svd_n_epochs": int(model.n_epochs),
        "svd_lr_pu": float(model.lr_pu),
        "svd_lr_qi": float(model.lr_qi),
        "svd_reg_pu": float(model.reg_pu),
        "svd_reg_qi": float(model.reg_qi),
    }
    with open("dataset_stats.pkl", "wb") as f:
        pickle.dump(dataset_stats, f)

    _ = hybrid_recommend(
        movies,
        cosine_sim,
        indices,
        model,
        ratings,
        user_id=1,
        movie_title="avatar",
        top_n=3,
    )

    with open("movies.pkl", "wb") as f:
        pickle.dump(movies, f)
    with open("cosine_sim.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)
    with open("indices.pkl", "wb") as f:
        pickle.dump(indices, f)
    with open("svd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"CV RMSE (mean ± std): {cv_metrics['test_rmse_mean']:.4f} ± {cv_metrics['test_rmse_std']:.4f}")
    print(f"Holdout test RMSE: {rmse:.4f}")
    print("✅ All models saved successfully!")


if __name__ == "__main__":
    train_all()
