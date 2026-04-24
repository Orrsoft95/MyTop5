
---
language: en
tags:
  - anime
  - recommendation-system
  - collaborative-filtering
  - content-based-filtering
  - hybrid
  - svd
license: mit
---

# MyTop5 — Anime Recommendation Engine Models

This repository contains serialized model files for the MyTop5 hybrid anime
recommendation system, built as a Data Analytics Capstone project.

## Model Files

| File | Description |
|---|---|
| `anime_metadata.pkl` | Cleaned anime DataFrame (titles, genres, synopsis, scores) |
| `content_feature_matrix.pkl` | Sparse TF-IDF + multi-hot genre feature matrix |
| `anime_index_map.pkl` | Dict mapping anime_id to feature matrix row index |
| `svd_model.pkl` | Trained Surprise SVD model (collaborative filtering) |
| `anime_titles.pkl` | Sorted list of anime titles for the Streamlit dropdown |

## How These Models Were Trained

- **Content feature matrix** — built from TF-IDF vectorization of anime synopses
  (5,000 max features, bigrams) combined with multi-hot encoded genre tags
- **SVD model** — trained on 20M+ MyAnimeList user ratings filtered to users
  with 50+ ratings and anime with 20+ ratings. Hyperparameters tuned via
  RandomizedSearchCV on a 25% sample, then retrained on the full dataset.

## Data Source

[Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
by dbdmobile on Kaggle.

## Usage

These models are loaded automatically by the
[MyTop5 Streamlit app](https://mytop5.streamlit.app) via `huggingface_hub`.

To load manually:
    
    import pickle
    from huggingface_hub import hf_hub_download
    
    path = hf_hub_download(
        repo_id="Orrsoft95/MyTop5",
        filename="svd_model.pkl",
        repo_type="model"
    )
    with open(path, "rb") as f:
        svd_model = pickle.load(f)
