# ============================================
# HYBRID MOVIE RECOMMENDATION SYSTEM
# ============================================

import pandas as pd
import ast

# ============================================
# CONTENT-BASED FILTERING (TMDB DATA)
# ============================================

# Load TMDB datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge movies and credits on title
movies = movies.merge(credits, on="title")

# Select only required columns
movies = movies[[
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]]

# --------------------------------------------
# Convert JSON-like columns into usable lists
# --------------------------------------------
def convert(text):
    try:
        return [i['name'].replace(" ", "") for i in ast.literal_eval(text)]
    except:
        return []

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert)

# --------------------------------------------
# Extract director from crew column
# --------------------------------------------
def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name'].replace(" ", "")
    except:
        return ""

movies["director"] = movies["crew"].apply(get_director)

# --------------------------------------------
# Clean missing values
# --------------------------------------------
movies["overview"] = movies["overview"].fillna("")
movies["director"] = movies["director"].fillna("")

# Remove unnecessary column
movies = movies.drop(columns=["crew"])

# --------------------------------------------
# Reduce noise (limit cast to top 3)
# --------------------------------------------
movies["cast"] = movies["cast"].apply(lambda x: x[:3])

# --------------------------------------------
# Convert lists to strings so that TF IDF vectoriser can use it
# --------------------------------------------
movies["genres"] = movies["genres"].apply(lambda x: " ".join(x))
movies["keywords"] = movies["keywords"].apply(lambda x: " ".join(x))
movies["cast"] = movies["cast"].apply(lambda x: " ".join(x))

# --------------------------------------------
# Create final feature column, one line per movie
# --------------------------------------------
movies["features"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["director"] + " " +
    movies["overview"]
)

# Convert to lowercase for consistency
movies["features"] = movies["features"].str.lower()

# ============================================
# PART 2: COLLABORATIVE FILTERING (MOVIELENS)
# ============================================

# Load ratings dataset (small version)
ratings = pd.read_csv("ratings_small.csv")

# Keep only required columns
ratings = ratings[["userId", "movieId", "rating"]]

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Fill missing values with 0
user_movie_matrix.fillna(0, inplace=True)

# ============================================
# PART 3: LINK BOTH DATASETS 
# ============================================

# Load links file to connect MovieLens ↔ TMDB
links = pd.read_csv("links_small.csv")

# Keep relevant columns
links = links[["movieId", "tmdbId"]]

# Remove missing values
links = links.dropna()

# Convert tmdbId to integer
links["tmdbId"] = links["tmdbId"].astype(int)

# Merge TMDB movies with MovieLens IDs
movies = movies.merge(links, left_on="movie_id", right_on="tmdbId")

# ============================================
# FINAL OUTPUT CHECK
# ============================================


# print("Movies Data Sample:")
# print(movies.head())

# print("\nUser-Movie Matrix Shape:")
# print(user_movie_matrix.shape)

# print("\nSetup Complete ✅")





# ============================================
# STEP 5: CONTENT-BASED FILTERING
# ============================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# 5.1 Convert text features into vectors (TF-IDF)
# --------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')

# Rows = movies
# Columns = words
# Values = importance of that word in that movie
# Apply TF-IDF on the 'features' column

tfidf_matrix = tfidf.fit_transform(movies['features'])

# --------------------------------------------
# 5.2 Compute similarity between all movies
# --------------------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --------------------------------------------
# 5.3 Reset index for easy lookup
# --------------------------------------------
movies = movies.reset_index(drop=True)

movies['title'] = movies['title'].str.lower()

# Create mapping: movie title -> index
# Creates a lookup table
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# --------------------------------------------
# 5.4 Recommendation Function
# --------------------------------------------
def recommend_movies(title, top_n=5):
    title = title.lower()
    
    if title not in indices:
        return "Movie not found in dataset"
    
    idx = indices[title]
    
    # Get similarity scores
    # enumerate gives us (index, similarity_score) for all movies compared to the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return movie titles
    return movies['title'].iloc[movie_indices].tolist()

# --------------------------------------------
# 5.5 Test the recommendation system
# --------------------------------------------
# print("\nRecommendations for Avatar:")
# print(recommend_movies("Avatar"))



# ============================================
# STEP 6: COLLABORATIVE FILTERING (SVD) Singular Value Decomposition
# ============================================

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# --------------------------------------------
# 6.1 Prepare data for Surprise library
# --------------------------------------------
#define the rating scale
reader = Reader(rating_scale=(0.5, 5))

#converting df into format that surprise understands
data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# --------------------------------------------
# 6.2 Split into train and test sets
# --------------------------------------------
trainset, testset = train_test_split(data, test_size=0.2)

# --------------------------------------------
# 6.3 Train SVD model
# --------------------------------------------
model = SVD()
model.fit(trainset)

# --------------------------------------------
# 6.4 Evaluate model
# --------------------------------------------
predictions = model.test(testset)

rmse = accuracy.rmse(predictions)

# Save RMSE for Streamlit
import pickle
with open("rmse.pkl", "wb") as f:
    pickle.dump(rmse, f)

# print("\nSVD Model Performance:")
# accuracy.rmse(predictions)

# --------------------------------------------
#  6.5 Predict rating for a user and a movie
#  --------------------------------------------

#  First, check available movieIds
# print(movies[['title', 'movieId']].head())

# getting movieId for "avatar"
# movie_id = movies[movies['title'] == 'avatar']['movieId'].values[0]

# Predict rating for user 1 on this movie
# pred = model.predict(uid=1, iid=movie_id)

# print("\nPredicted rating for user 1 on Avatar:")
# print(pred.est)



# ============================================
# STEP 7: HYBRID RECOMMENDATION SYSTEM
# ============================================

# --------------------------------------------
# 7.1 Hybrid Recommendation Function
# --------------------------------------------
def hybrid_recommend(user_id, movie_title, top_n=5):
    movie_title = movie_title.lower()
    
    if movie_title not in indices:
        return "Movie not found"
    
    # 🔹 Check if user exists (cold-start handling)
    known_users = set(ratings['userId'].unique())
    is_new_user = user_id not in known_users
    
    # Get index of the movie
    idx = indices[movie_title]
    
    # Get similarity scores (CBF)
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Take top 20 similar movies
    sim_scores = sim_scores[1:20]
    
    hybrid_scores = []
    
    for i, sim in sim_scores:
        movie_id = movies.iloc[i]['movieId']
        
        if is_new_user:
            # 🔹 Cold-start → use only content-based similarity
            final_score = sim
        else:
            # 🔹 Collaborative filtering (SVD)
            pred_rating = model.predict(user_id, movie_id).est
            
            # Hybrid score
            final_score = (0.6 * pred_rating) + (0.4 * sim)
        
        hybrid_scores.append((i, final_score))
    
    # Sort by final score
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N movies
    top_movies = hybrid_scores[:top_n]
    
    movie_indices = [i[0] for i in top_movies]
    
    return movies['title'].iloc[movie_indices].tolist()


# --------------------------------------------
# 7.2 Test Hybrid Recommendation
# --------------------------------------------
# print("\nHybrid Recommendations for User 1 (based on Avatar):")
# print(hybrid_recommend(user_id=1, movie_title="Avatar"))


# ============================================
# SAVE PRECOMPUTED OBJECTS
# ============================================

with open("movies.pkl", "wb") as f:
    pickle.dump(movies, f)

with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)

with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ All models saved successfully!")