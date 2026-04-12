# ============================================
# STREAMLIT APP
# ============================================

import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from recommender_core import hybrid_recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Hybrid Movie Recommendation System")
st.markdown("Predictive analytics: TF-IDF + cosine similarity, Surprise SVD, normalized hybrid fusion.")


@st.cache_resource
def load_models():
    with open("movies.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    with open("svd_model.pkl", "rb") as f:
        model = pickle.load(f)
    ratings = pd.read_csv("ratings_small.csv")
    with open("rmse.pkl", "rb") as f:
        rmse = pickle.load(f)

    dataset_stats = None
    cv_metrics = None
    try:
        with open("dataset_stats.pkl", "rb") as f:
            dataset_stats = pickle.load(f)
    except OSError:
        pass
    try:
        with open("cv_metrics.pkl", "rb") as f:
            cv_metrics = pickle.load(f)
    except OSError:
        pass

    return movies, cosine_sim, indices, model, ratings, rmse, dataset_stats, cv_metrics


movies, cosine_sim, indices, model, ratings, rmse, dataset_stats, cv_metrics = load_models()

if "results" not in st.session_state:
    st.session_state.results = None


with st.expander("Dataset and methodology (for report)", expanded=False):
    if dataset_stats is not None:
        st.subheader("Dataset summary")
        st.dataframe(
            pd.DataFrame([dataset_stats]).T.rename(columns={0: "value"}),
            use_container_width=True,
        )
    else:
        st.info("Run `python train.py` to generate `dataset_stats.pkl`.")

    if cv_metrics is not None:
        st.subheader("Cross-validated RMSE (3 folds)")
        st.write(
            f"Mean: **{cv_metrics['test_rmse_mean']:.4f}**  "
            f"Std: **{cv_metrics['test_rmse_std']:.4f}**"
        )
    else:
        st.info("Run `python train.py` to generate `cv_metrics.pkl`.")

    st.subheader("Hybrid score (warm users)")
    st.markdown(
        "Within the cosine neighborhood, **predicted rating** (SVD) and **cosine similarity** "
        "are min–max scaled to [0, 1], then combined as **0.6 × pred_norm + 0.4 × sim_norm**. "
        "New users (no ratings) are ranked by **similarity only**."
    )

    st.subheader("Single-split test RMSE")
    st.metric("SVD on 20% holdout", f"{float(rmse):.4f}")


st.sidebar.header("Settings")
user_id = st.sidebar.number_input("User ID", min_value=1, step=1, format="%d")


def run_recommend(seed_title: str) -> None:
    st.session_state.results = hybrid_recommend(
        movies,
        cosine_sim,
        indices,
        model,
        ratings,
        int(user_id),
        seed_title,
        top_n=5,
    )


movie_list = sorted(movies["title"].unique())
selected_movie = st.selectbox("Select a seed movie", movie_list)

col1, col2 = st.columns(2)

with col1:
    if st.button("Recommend", type="primary"):
        run_recommend(selected_movie)

    if st.session_state.results is not None:
        st.subheader("Recommendations")
        display_df = st.session_state.results.copy()
        display_df["title"] = display_df["title"].str.title()
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

with col2:
    if st.button("Surprise me"):
        random_movie = random.choice(movie_list)
        st.session_state.surprise_pick = random_movie
        run_recommend(random_movie)
    if getattr(st.session_state, "surprise_pick", None):
        st.caption(f"Last random seed: **{st.session_state.surprise_pick.title()}**")


st.subheader("Figures for analysis")

c1, c2 = st.columns(2)

with c1:
    if st.checkbox("Rating values (histogram)", value=True):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(ratings["rating"], bins=10, color="steelblue", edgecolor="white")
        ax.set_title("Distribution of ratings")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if st.checkbox("Ratings per user (long tail)"):
        counts = ratings.groupby("userId").size()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(counts, bins=30, color="coral", edgecolor="white")
        ax.set_title("Number of ratings per user")
        ax.set_xlabel("Ratings count")
        ax.set_ylabel("Users")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with c2:
    if st.checkbox("Cosine similarities for seed movie (all others)"):
        idx = int(indices[selected_movie])
        row = cosine_sim[idx]
        mask = row < 0.9999
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(row[mask], bins=40, color="seagreen", edgecolor="white")
        ax.set_title(f"Cosine sim vs other movies ({selected_movie.title()})")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Movies")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

if st.session_state.results is not None:
    res = st.session_state.results
    c3, c4 = st.columns(2)
    with c3:
        if st.checkbox("Similarity (top picks)"):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(res["title"].str.title(), res["similarity"], color="teal")
            ax.set_title("Content: cosine similarity")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with c4:
        if st.checkbox("Hybrid final score (top picks)"):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(res["title"].str.title(), res["final_score"], color="slateblue")
            ax.set_title("Hybrid score (after normalization)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if st.checkbox("Predicted rating (SVD) for top picks") and res["predicted_rating"].notna().any():
        fig, ax = plt.subplots(figsize=(5, 3))
        sub = res.dropna(subset=["predicted_rating"])
        ax.barh(sub["title"].str.title(), sub["predicted_rating"], color="orange")
        ax.set_title("Collaborative: predicted rating")
        ax.set_xlabel("Predicted rating (0.5–5)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
