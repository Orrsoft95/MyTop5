"""
preprocess.py
------------
Loads raw Kaggle CSVs, cleans & transforms them,
and then serializes the outputs needed by the recommendation
engine to models/

Outputs
------------
anime_metadata.pkl          : cleaned anime dataframe
content_feature_matrix.pkl  : combined TF-IDF + genre matrix
anime_index_map.pkl         : dict mapping anime_id to matrix row index
svd_model.pkl               : trained SVD model
anime_titles.pkl            : sorted list of titles for use in streamlit dropdown (select your top 5)

Run
---
python src/preprocess.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import SVD, Dataset, Reader

#Random state to be used for preprocessing
RAND_STATE = 42

# Establish file paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

ANIME_CSV = os.path.join(DATA_DIR, "anime-filtered.csv")
USERS_CSV = os.path.join(DATA_DIR, "users-score-2023.csv")

# Tuneable parameters
MIN_RATINGS_PER_USER = 50 #Drop users w/ fewer ratings than this
MIN_RATINGS_PER_ANIME = 20 #Drop anime w/ fewer ratings than this
TFIDF_MAX_FEATURES = 5000 #Vocabulary cap for synopsis vectorizer
CONTENT_WEIGHT = 0.6 #Will be used downstream in hybrid.py
COLLAB_WEIGHT = 0.4