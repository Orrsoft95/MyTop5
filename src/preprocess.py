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

# 1: Load & Clean anime metadata!

def load_anime(path: str) -> pd.DataFrame:
    """
    Load anime-filtered.csv and return a cleaned dataframe.

    Only keeps rows w/ a valid synopsis & at least ONE genre tag.
    Standardizes column names to snake_case
    """

    print("Loading anime metadata...")
    df = pd.read_csv(path)
    print(f"{len(df):,} anime present BEFORE cleaning")

    #Normalize column names! trim, lowercase text, replace spaces w/ underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    #"synopsis" is misspelled as "sypnopsis" in anime-filtered.csv; correct that
    df.rename(columns={"sypnopsis": "synopsis"}, inplace=True)

    """
    Drop rows missing the fields needed for content-based filtering:
    anime_id, name, genres, synopsis
    """
    required_cols = ["anime_id", "name", "genres", "synopsis"]
    df.dropna(subset=required_cols, inplace=True)

    #Drop rows where synopsis or genres is an empty string/whitespace
    df = df[df["synopsis"].str.strip() != ""]
    df = df[df["genres"].str.strip() != ""]

    #Reset index!
    df.reset_index(inplace=True, drop=True)

    print(f"{len(df):,} anime present AFTER cleaning")

    return df

def build_content_matrix(df: pd.DataFrame):
    """
    Constructs a combined sparse feature matrix from:
        -TF-IDF vectors of synopsis text
        -Multi-hot encoded genre tags

    Returns
    ---------
    feature_matrix  : scipy sparse matrix in shape (n_anime, n_features)
    anime_index     : dict {anime_id, row_index}
    """

    print("Building TF-IDF synopsis matrix...")

    tfidf = TfidfVectorizer(
        max_features= TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),  #Include unigrams & bigrams
        min_df=2, #Ignore terms that appear less than twice
    )

    #fit the vectorizer to our synopses
    tfidf_matrix = tfidf.fit_transform(df["synopsis"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("Now building multi-hot genre matrix...")

    """
    Genres are stored in anime-filtered.csv as comma-separated strings;
    split these strings into lists for processing
    """ 
    genre_lists = df["genres"].apply(
        lambda x: sorted(
            [g.strip() for g in x.split(",") if g.strip()]
            )
    )

    #fit our multi-label binarizer to our genre_lists & create a sparse row matrix out of it 
    mlb = MultiLabelBinarizer()
    genre_matrix = csr_matrix(mlb.fit_transform(genre_lists))
    print(f"Genre matrix shape: {genre_matrix.shape}")
    print(f"Unique genres found: {len(mlb.classes_)}")

    #HORIZONTALLY stack both matrices into one combined feature matrix!
    feature_matrix = hstack([tfidf_matrix, genre_matrix], format="csr")
    print(f"Combined feature matrix shape: {feature_matrix.shape}")

    #Map anime_id > row index for fast lookup at inference time
    anime_index_map = {
        anime_id: idx for idx, anime_id in enumerate(df["anime_id"])
    }

    return feature_matrix, anime_index_map