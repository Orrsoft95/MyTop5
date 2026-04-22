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
from surprise.model_selection import RandomizedSearchCV

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

def load_ratings(path: str, anime_ids: set) -> pd.DataFrame:
    """
    Load users-score-2023.csv, apply data quality filters, and return
    a cleaned, long-format ratings dataframe w/ columns:
    user_id, anime_id, rating

    FILTERS APPLIED
    ---------------
    - Only keep anime that are present in our cleaned anime dataframe
    - Drop users with < MIN_RATINGS_PER_USER
    - Drop anime with fewer than MIN_RATINGS_PER_ANIME
    """

    print("Loading user ratings (this may take a moment)...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print(f"Raw ratings loaded: {len(df):,} rows")

    #Only keep ratings for anime that exist in our cleaned dataset
    df = df[df["anime_id"].isin(anime_ids)]
    print(f"# of records after anime filtering: {len(df):,} rows")

    #Drop ratings of 0 (0 = "not rated" in MAL)
    df = df[df["rating"] > 0]
    print(f"# of records after removing unscored entries: {len(df):,} rows")

    #Filter out users w/ too few ratings
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    df = df[df["user_id"].isin(valid_users)]
    print(f"# of records after USER filter (>={MIN_RATINGS_PER_USER} ratings): {len(df):,} rows")

    #Filter out anime with too few ratings
    anime_counts = df["anime_id"].value_counts()
    valid_anime = anime_counts[anime_counts >= MIN_RATINGS_PER_ANIME].index
    df = df[df["anime_id"].isin(valid_anime)]
    print(f"# of records after ANIME filter (>={MIN_RATINGS_PER_ANIME} ratings): {len(df):,} rows")

    #Ensure we've only kept necessary columns!
    df = df[["user_id", "anime_id", "rating"]].reset_index(drop=True)

    print(f"Unique USERS: {df['user_id'].nunique():,}")
    print(f"Unique ANIME: {df['anime_id'].nunique():,}")

    return df

def train_svd(ratings_df: pd.DataFrame) -> SVD:
    """
    Trains a surprise SVD model on the (filtered) ratings dataframe.
    Uses RandomizedSearchCV to efficiently tune hyperparameters.
    Best parameters are selected by lowest root mean square err via 3-fold cross-validation.
    Rating scale is inferred from the data, but is 1-10 in the case of MAL).
    """

    print("Training our SVD model...")
    min_r = ratings_df["rating"].min()
    max_r = ratings_df["rating"].max()

    reader = Reader(rating_scale=(min_r, max_r))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "anime_id", "rating"]],
        reader
    )

    #parameters to test for best fit:
    param_distributions = {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30, 50],
        "lr_all": [0.005, 0.01, 0.15],
        "reg_all": [0.02, 0.05, 0.75]
    }


    print("Hyperparameter tuning now...")
    #Hyperparameter tuning:
    rs = RandomizedSearchCV(
        SVD,
        param_distributions,
        measures=["rmse"],
        cv=3,
        n_iter=25, #Only try 25 random combinations instead of all 81
        refit=True,
        random_state=RAND_STATE
        joblib_verbose=1,
        n_jobs=-2 #Use all but 1 CPU core
    )

    print("Fitting the model to our data...")
    rs.fit(data)

    best_params = rs.best_params["rmse"]
    best_rmse = rs.best_score["rmse"]
    print(f"BEST RMSE: {best_rmse:.4f}")
    print(f"BEST PARAMETERS: {best_params}")

    #Since refit was set to True in GridSearchCV, gs.best_estimator is already fitted on our full dataset
    svd = rs.best_estimator["rmse"]
    return svd


def pickle_save(obj, filename:str):
    """
    Serialize obj to models/[filename]
    """
    path = os.path.join(MODELS_DIR, filename)
    print("Saving {obj} to pickle file...")
    with open(path, "wb") as file:
        pickle.dump(obj, file)

    print(f"Successfully saved model to models/{filename}")
    
#MAIN PIPELINE!
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    #Step 1 - load anime metadata
    anime_df = load_anime(ANIME_CSV)

    #Step 2 - build content feature matrix
    feature_matrix, anime_index_map = build_content_matrix(anime_df)

    #Step 3 - Load user ratings
    ratings_df = load_ratings(USERS_CSV, anime_ids=set(anime_df["anime_id"]))

    #Step 4 - Train SVD model on ratings dataframe
    svd_model = train_svd(ratings_df)

    #Step 5 - create a sorted title list for streamlit app dropdown
    anime_titles = sorted(anime_df["name"].dropna().unique().tolist())

    #Step 6 - save all necessary outputs to pickle files!
    pickle_save(anime_df, "anime_metadata.pk1")
    pickle_save(feature_matrix, "content_feature_matrix.pk1")
    pickle_save(anime_index_map, "anime_index_map.pk1")
    pickle_save(svd_model, "svd_model.pk1")
    pickle_save(anime_titles, "anime_titles.pk1")

    print("\nPreprocessing complete! All outputs have been saved to models/.")
    print(f"Anime in catalog: {len(anime_df):,}")
    print(f"Ratings retained: {len(ratings_df):,}")
    print(f"Feature matrix: {feature_matrix.shape}")

if __name__ == "__main__":
    main()