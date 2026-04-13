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
