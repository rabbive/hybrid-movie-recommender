# Predictive Analytics Report: Hybrid Movie Recommender

This document summarizes methodology, data, and evaluation for course submission. It mirrors the implementation in `recommender.py`, `recommender_core.py`, and `app.py`.

## 1. Problem

Predict which movies a user may prefer given **sparse explicit ratings** (MovieLens) and **rich text metadata** (TMDB: genres, keywords, cast, director, overview). We combine **content-based** similarity with **collaborative** latent-factor predictions and handle **cold-start** users who have no rating history.

## 2. Data

- **MovieLens-small** (`ratings_small.csv`): `userId`, `movieId`, `rating` on a 0.5–5 scale.
- **TMDB** (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`): metadata merged on `title`.
- **Links** (`links_small.csv`): maps MovieLens `movieId` to TMDB `tmdbId` so ratings align with content rows.

After the join, each row has both MovieLens `movieId` and a single text `features` field used for TF-IDF.

**Descriptive statistics** (exact values after training are saved in `dataset_stats.pkl` and shown in the Streamlit “Dataset & methodology” section): number of users, movies, ratings, matrix **sparsity** \(1 - \frac{\text{ratings}}{|\text{users}| \times |\text{movies}|}\), and rating scale.

## 3. Content-based model: TF-IDF and cosine similarity

- We concatenate genre, keyword, cast, director, and overview text into `features` and apply **TF-IDF** (`TfidfVectorizer`, English stop words).
- Each movie becomes a sparse high-dimensional vector. **Cosine similarity** between vectors measures how aligned two movies are in vocabulary space (angle, not Euclidean distance).
- For a seed movie, we take the top neighbors by cosine similarity (excluding the seed) as the **candidate pool** for hybrid ranking.

**Figure for the report:** rating distribution histogram; optional histogram of cosine similarities for one movie’s similarity row (available in the app).

## 4. Collaborative model: SVD (Surprise)

- We use the **Surprise** library’s `SVD`, a matrix factorization method that learns latent user and item factors from observed ratings.
- **Training:** 80% train / 20% test split on ratings; RMSE on the **held-out test** set measures average prediction error in rating units.
- **Robustness:** **3-fold cross-validation** on the full rating matrix reports mean ± standard deviation of test RMSE across folds (`cv_metrics.pkl`).

**Table for the report:** list SVD hyperparameters (`n_factors`, `n_epochs`, `lr_pu`, `lr_qi`, `reg_pu`, `reg_qi`, etc.) from `dataset_stats.pkl` after training (Surprise stores per-term learning rates and regularizers).

## 5. Hybrid fusion and cold-start

- **Warm user** (appears in `ratings_small.csv`): for each candidate neighbor we have a **predicted rating** \(\hat{r}\) from SVD and a **cosine similarity** \(s\) in \([0,1]\) (cosine output).
- Because \(\hat{r}\) and \(s\) live on different scales, we **min–max normalize** \(\hat{r}\) and \(s\) **within the candidate neighborhood** to \([0,1]\), then:

\[
\text{final} = 0.6 \cdot \hat{r}_{\text{norm}} + 0.4 \cdot s_{\text{norm}}
\]

Weights \(0.6 / 0.4\) emphasize collaborative signal while keeping content diversity; document this as a **design choice** (tunable in code via `cf_weight` / `content_weight` in `recommender_core.hybrid_recommend`).

- **Cold-start user** (no ratings): we rank by **cosine similarity only** (`final_score = s`), since SVD has no reliable user latent vector.

**Figures for the report:** horizontal bar charts of similarity and hybrid scores for the top recommendations (Streamlit checkboxes).

## 6. Results and limitations

- Report **CV RMSE (mean ± std)** and **single-split test RMSE** from the training run.
- Show one **worked example**: pick `userId`, seed movie, table with title, `similarity`, `predicted_rating`, normalized columns if shown, and `final_score`.

**Limitations:** small public sample vs production scale; random train/test split variance; linear fusion is simple; dense full pairwise cosine matrix is memory-heavy for very large catalogs.

## 7. Reproducibility

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py             # writes .pkl artifacts
streamlit run app.py
```

Use **Python 3.10–3.12**; `numpy>=2` can break older `scikit-surprise` wheels—`requirements.txt` pins `numpy<2`.
