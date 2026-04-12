# ============================================
# STREAMLIT APP
# ============================================

import streamlit as st
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("Fast AI-powered recommendations ⚡")

# --------------------------------------------
# LOAD PRECOMPUTED DATA
# --------------------------------------------
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
    
    with open("ratings_small.csv", "r") as f:
        ratings = pd.read_csv(f)

    with open("rmse.pkl", "rb") as f:
        rmse = pickle.load(f)

    return movies, cosine_sim, indices, model, ratings, rmse

movies, cosine_sim, indices, model, ratings, rmse = load_models()

if "results" not in st.session_state:
    st.session_state.results = None



# --------------------------------------------
# HYBRID FUNCTION
# --------------------------------------------
def hybrid_recommend(user_id, movie_title, top_n=5):
    movie_title = movie_title.lower()

    if movie_title not in indices:
        return None

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:20]

    # 🔹 Check if user exists
    known_users = set(ratings['userId'].unique())
    is_new_user = user_id not in known_users

    data = []

    for i, sim in sim_scores:
        title = movies.iloc[i]['title']

        if is_new_user:
            # 🔹 Cold-start → only content-based
            pred = None
            final_score = sim
        else:
            movie_id = movies.iloc[i]['movieId']
            pred = model.predict(user_id, movie_id).est
            final_score = (0.6 * pred) + (0.4 * sim)

        data.append({
            "title": title,
            "similarity": sim,
            "predicted_rating": pred,
            "final_score": final_score
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by="final_score", ascending=False).head(top_n)

    return df
# --------------------------------------------
# UI
# --------------------------------------------
st.sidebar.header("⚙️ Settings")

user_id = st.sidebar.number_input(
    "User ID",
    min_value=1,
    step=1,
    format="%d"
)

movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("🎥 Select a Movie", movie_list)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Recommend"):
        st.session_state.results = hybrid_recommend(user_id, selected_movie)

    if st.session_state.results is not None:
        st.subheader("✨ Recommendations")

        for i, row in st.session_state.results.iterrows():
            st.write(f"🎬 {row['title'].title()}")

with col2:
    if st.button("🎲 Surprise Me"):
        random_movie = random.choice(movie_list)
        st.write(f"Try: **{random_movie.title()}**")


# --------------------------------------------
# MODEL INSIGHTS UI (FIXED LAYOUT)
# --------------------------------------------
st.subheader("📊 Model Insights")

col1, col2 = st.columns(2)

# 🔹 Ratings Distribution
with col1:
    if st.checkbox("Show Rating Distribution"):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(ratings['rating'], bins=10)
        ax.set_title("Ratings Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# 🔹 RMSE
with col2:
    if st.checkbox("Show Model Performance"):
        st.metric(label="SVD Model RMSE", value=round(rmse, 4))

if st.session_state.results is not None:

    if st.checkbox("Show Similarity Graph"):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(
            st.session_state.results['title'].str.title(),
            st.session_state.results['similarity']
        )
        ax.set_title("Similarity Scores")

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    if st.checkbox("Show Hybrid Score Breakdown"):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(
            st.session_state.results['title'].str.title(),
            st.session_state.results['final_score']
        )
        ax.set_title("Hybrid Scores")

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)