"""
FastAPI backend for the hybrid movie recommender.
Run with: uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recommender_core import hybrid_recommend

_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in [
        ("movies", "movies.pkl"),
        ("cosine_sim", "cosine_sim.pkl"),
        ("indices", "indices.pkl"),
        ("model", "svd_model.pkl"),
    ]:
        with open(path, "rb") as f:
            _models[name] = pickle.load(f)

    _models["ratings"] = pd.read_csv("ratings_small.csv")

    for name, path in [
        ("rmse", "rmse.pkl"),
        ("dataset_stats", "dataset_stats.pkl"),
        ("cv_metrics", "cv_metrics.pkl"),
    ]:
        try:
            with open(path, "rb") as f:
                _models[name] = pickle.load(f)
        except OSError:
            _models[name] = None

    yield
    _models.clear()


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/movies")
async def list_movies():
    movies = _models["movies"]
    titles = sorted(movies["title"].str.title().unique().tolist())
    return {"movies": titles}


@app.get("/recommend")
async def recommend(user_id: int, movie_title: str, top_n: int = 5):
    result = hybrid_recommend(
        _models["movies"],
        _models["cosine_sim"],
        _models["indices"],
        _models["model"],
        _models["ratings"],
        user_id=user_id,
        movie_title=movie_title.lower(),
        top_n=top_n,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found.")
    records = result.where(result.notna(), other=None)
    return records.to_dict(orient="records")


@app.get("/stats")
async def get_stats():
    rmse = _models.get("rmse")
    return {
        "dataset_stats": _models.get("dataset_stats"),
        "cv_metrics": _models.get("cv_metrics"),
        "rmse": float(rmse) if rmse is not None else None,
    }


@app.get("/charts/ratings-distribution")
async def ratings_distribution():
    """Histogram data for the distribution of all ratings."""
    ratings = _models["ratings"]
    counts, bin_edges = pd.cut(ratings["rating"], bins=10, retbins=True)
    hist = ratings["rating"].value_counts().sort_index()
    return [
        {"rating": float(r), "count": int(c)}
        for r, c in hist.items()
    ]


@app.get("/charts/ratings-per-user")
async def ratings_per_user():
    """Histogram data for number of ratings each user has given."""
    import numpy as np

    ratings = _models["ratings"]
    user_counts = ratings.groupby("userId").size().values
    counts, bin_edges = np.histogram(user_counts, bins=30)
    return [
        {"bin_start": round(float(bin_edges[i]), 1), "bin_end": round(float(bin_edges[i + 1]), 1), "count": int(counts[i])}
        for i in range(len(counts))
    ]


@app.get("/charts/cosine-similarities")
async def cosine_similarities(movie_title: str):
    """Histogram of cosine similarities between a seed movie and all others."""
    import numpy as np

    indices = _models["indices"]
    cosine_sim = _models["cosine_sim"]
    title_lower = movie_title.lower()
    if title_lower not in indices:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found.")
    idx = int(indices[title_lower])
    row = cosine_sim[idx]
    mask = row < 0.9999
    values = row[mask]
    counts, bin_edges = np.histogram(values, bins=40)
    return [
        {"bin_start": round(float(bin_edges[i]), 4), "bin_end": round(float(bin_edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(len(counts))
    ]
