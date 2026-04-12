"""
Shared hybrid recommendation logic for training (recommender.py) and Streamlit (app.py).
"""

from __future__ import annotations

import pandas as pd


def hybrid_recommend(
    movies: pd.DataFrame,
    cosine_sim,
    indices: pd.Series,
    model,
    ratings: pd.DataFrame,
    user_id: int,
    movie_title: str,
    top_n: int = 5,
    neighborhood: int = 19,
    cf_weight: float = 0.6,
    content_weight: float = 0.4,
) -> pd.DataFrame | None:
    """
    Content: cosine neighbors on TF-IDF vectors. Collaborative: Surprise SVD predictions.
    Warm users: min-max normalize predicted rating and similarity within the candidate
    neighborhood, then combine cf_weight * pred_norm + content_weight * sim_norm.
    Cold-start (user not in ratings): final_score = cosine similarity only.
    """
    movie_title = movie_title.lower()

    if movie_title not in indices:
        return None

    known_users = set(ratings["userId"].unique())
    is_new_user = user_id not in known_users

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : neighborhood + 1]

    rows = []
    for i, sim in sim_scores:
        title = movies.iloc[i]["title"]
        movie_id = movies.iloc[i]["movieId"]

        if is_new_user:
            pred = None
            pred_norm = None
            sim_norm = None
            final_score = float(sim)
        else:
            pred = float(model.predict(user_id, movie_id).est)
            pred_norm = None
            sim_norm = None
            final_score = None  # filled after normalization

        rows.append(
            {
                "title": title,
                "movieId": int(movie_id),
                "similarity": float(sim),
                "predicted_rating": pred,
                "pred_norm": pred_norm,
                "sim_norm": sim_norm,
                "final_score": final_score,
            }
        )

    df = pd.DataFrame(rows)

    if not is_new_user:
        preds = df["predicted_rating"].astype(float)
        sims = df["similarity"].astype(float)
        p_min, p_max = preds.min(), preds.max()
        s_min, s_max = sims.min(), sims.max()
        eps = 1e-8
        df["pred_norm"] = (preds - p_min) / (p_max - p_min + eps)
        df["sim_norm"] = (sims - s_min) / (s_max - s_min + eps)
        df["final_score"] = cf_weight * df["pred_norm"] + content_weight * df["sim_norm"]

    df = df.sort_values(by="final_score", ascending=False).head(top_n)
    return df.reset_index(drop=True)
