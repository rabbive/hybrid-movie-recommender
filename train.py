#!/usr/bin/env python3
"""Train models and write pickles (movies, cosine_sim, indices, svd_model, rmse, cv_metrics, dataset_stats)."""

from recommender import train_all

if __name__ == "__main__":
    train_all()
