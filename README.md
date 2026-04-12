# 🎬 Hybrid Movie Recommendation System

## 📌 Overview

This project implements a **Hybrid Movie Recommendation System** that combines both **Collaborative Filtering** and **Content-Based Filtering** techniques to provide accurate and personalized movie recommendations.

The system leverages user ratings and movie metadata to suggest movies that match user preferences effectively.

---

## 🚀 Features

* 🔍 Search movies by title
* ⭐ Predict user ratings for movies
* 🎯 Personalized recommendations
* 🤝 Hybrid approach (Content + Collaborative)
* 📊 Uses SVD (Singular Value Decomposition) for collaborative filtering
* 🧠 Uses TF-IDF + Cosine Similarity for content-based filtering
* 🌐 Optional Streamlit-based UI

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Surprise Library (SVD Model)
* Streamlit (for UI)

---

## 📂 Project Structure

```
├── app.py                 # Streamlit frontend
├── recommender.py               # Recommendation logic
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
│   ratings_small.csv
├── links_small.csv
└── README.md
```

---

## ⚙️ How It Works

### 1. Content-Based Filtering

* Uses movie metadata such as genres, keywords, etc.
* Applies **TF-IDF Vectorization**
* Computes similarity using **Cosine Similarity**

### 2. Collaborative Filtering

* Uses user ratings dataset
* Implements **SVD (Singular Value Decomposition)**
* Learns latent features of users and movies

### 3. Hybrid Approach

* Combines results from both methods
* Improves recommendation accuracy
* Balances personalization and similarity

---

## ▶️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/hybrid-movie-recommender.git
cd hybrid-movie-recommender
```


### 2. Run the application

```
python recoomender.py
streamlit run app.py
```

---

## 📊 Model Performance

* RMSE (Collaborative Filtering): ~0.89
* Provides top-N recommendations based on similarity and predicted ratings

---


## 🔮 Future Improvements

* Add deep learning models
* Improve UI/UX
* Deploy on cloud (AWS / Heroku)
* Add user authentication

---

## 👨‍💻 Author

* nando-g

---

## 📜 License

This project is for educational purposes only.
