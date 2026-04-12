# Hybrid Movie Recommendation System

## Overview

Hybrid recommender for a **predictive analytics** course project: **content-based** recommendations use **TF-IDF** on TMDB text features and **cosine similarity** between movies; **collaborative** predictions use **Surprise SVD** on MovieLens ratings. The **Streamlit** app (`app.py`) loads trained artifacts and explains methodology with tables and plots. Full narrative for submission lives in [REPORT.md](REPORT.md).

## How the hybrid score works (warm users)

Within each seed movie’s cosine neighborhood, **predicted rating** (SVD, roughly 0.5–5) and **cosine similarity** (0–1) are **min–max normalized** to \([0,1]\), then combined:

`final = 0.6 * pred_norm + 0.4 * sim_norm`

**Cold-start** users (no rows in `ratings_small.csv`) are ranked by **cosine similarity only**. Implementation: [recommender_core.py](recommender_core.py).

## Project structure

```
├── app.py                 # Streamlit UI + figures
├── recommender.py         # Training pipeline (run via train.py or directly)
├── recommender_core.py    # Shared hybrid_recommend()
├── train.py               # One-command training → pickles
├── requirements.txt       # Pins numpy<2 for scikit-surprise
├── REPORT.md              # Professor-facing methodology write-up
├── ratings_small.csv
├── links_small.csv
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── README.md
```

After training, the repo root also contains `movies.pkl`, `cosine_sim.pkl`, `indices.pkl`, `svd_model.pkl`, `rmse.pkl`, `dataset_stats.pkl`, and `cv_metrics.pkl` (large binaries; regenerate locally—do not commit if your course forbids large files).

## Technologies

- Python 3.10–3.12 (3.14 not supported by scikit-surprise at time of writing)
- pandas, NumPy (see `requirements.txt`: pin below 2.x for scikit-surprise), scikit-learn, scikit-surprise, Streamlit, matplotlib

## Setup

```bash
git clone https://github.com/rabbive/hybrid-movie-recommender.git
cd hybrid-movie-recommender
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py             # builds all .pkl artifacts
streamlit run app.py
```

## Evaluation

- **3-fold cross-validated RMSE** (mean ± std) is saved to `cv_metrics.pkl` and shown in the app.
- **Single 80/20 split** RMSE on the saved model is in `rmse.pkl` (typical value ~0.89).

## Model performance (example run)

Figures vary slightly with randomness; see your `train.py` stdout and Streamlit metrics after you train.

## Authors

- Original: nando-g
- Fork / coursework: rabbive

## License

Educational use only.
