"""
content_filter.py
-----------------
Creates the content-based filtering for the anime recommendation model.

Given a list of anime titles selected by the user, computes a centroid feature
vector from their combined content features, and returns the most similar anime
from the catalog using cosine similarity.

Inputs (from models/)
---------------------
content_feature_matrix.pkl      : sparse TF-IDF + genre feature matrix
anime_index_map.pkl             : dict mapping anime_id to matrix row index
anime_metadata.pkl              : cleaned anime dataframe

Public API
---------------
get_content_recommendations(
    selected_titles : list[str],
    anime_df        : pd.DataFrame,
    feature_matrix  : scipy sparse matrix,
    anime_index_map : dict,
    top_n           : int = 50
) -> pd.DataFrame
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

#Internal helper functions!

def _titles_to_ids(selected_titles: list[str], anime_df: pd.DataFrame,) -> list[int]:
    """
    Convert a list of anime title strings to their corresponding anime_ids.

    Matching is case-insensitive & strips whitespace; titles that can't be matched
    are skipped with a warning instead of raising an exception, so a partial match
    will still produce recommendations.

    Parameters
    ------------
    selected_titles : list of title strings (to be selected via dropdown)
    anime_df        : cleaned anime metadata dataframe

    Returns
    ------------
    list of matched anime_id integers
    """

    # Build a LOWERCASE title > anime_id lookup for fast matching
    title_to_id = {
        title.strip().lower(): anime_id
        for title, anime_id in zip(anime_df["name"], anime_df["anime_id"])
    }

    matched_ids = []

    for title in selected_titles:
        key = title.strip().lower()
        if key in title_to_id:
            matched_ids.append(title_to_id[key])
        else:
            print(f"WARNING: '{title} not found in catalog - this title will be skipped.'")
    
    return matched_ids

def _build_centroid(anime_ids: list[int], feature_matrix: csr_matrix, anime_index_map: dict,) -> np.ndarray:
    """
    Compute the centroid (mean) feature vector for a list of anime_ids.

    The centroid represents the "average" content profile of the user's selected titles, & is used
    as the query vector for cosine similarity.

    Parameters
    ------------
    anime_ids       : list of anime_id integers to average on
    feature_matrix  : sparse feature matrix (n_anime x n_features)
    anime_index_map : dict mapping anime_id to matrix row index

    Returns
    ------------
    centroid: np.ndarray of shape (1, n_features)
    """

    row_indices = []
    for anime_id in anime_ids:
        idx = anime_index_map.get(anime_id)
        if idx is not None:
            row_indices.append(idx)
        else:
            print(f"WARNING: anime_id {anime_id} not in feature matrix - this id will be skipped.")
    
    #Raise an error if NONE of the anime_ids are found in feature_matrix
    if not row_indices:
        raise ValueError(
            "ERROR: None of the selected anime could be found in the feature matrix."
            "Check that preprocess.py has been run & all models are up to date."
        )
    
    #Extract rows for selected anime & compute the mean vector
    selected_vectors = feature_matrix[row_indices]
    centroid = selected_vectors.mean(axis=0)

    #Return centroid as an array instead of numpy matrix
    return np.asarray(centroid)

def get_content_recommendations(
        selected_titles: list[str],
        anime_df: pd.DataFrame,
        feature_matrix: csr_matrix,
        anime_index_map: dict,
        top_n: int=50,
) -> pd.DataFrame:
    """
    Return the top_n most content-similar anime for a given list of titles.

    The function computes a centroid vector from the user's selected titles & ranks all other anime
    by cosine similarity to that centroid.
    Selected titles are excluded from the results.

    Parameters
    ------------
    selected_titles     : list of anime title strings (to be selected via dropdown)
    anime_df            : cleaned anime metadata dataframe
    feature_matrix      : sparse TF-IDF + genre feature matrix
    anime_index_map     : dict mapping anime_id to matrix row index
    top_n               : number of candidates to return (default 50, wil be narrowed further by hybrid.py)

    Returns
    ------------
    pd.DataFrame with columns:
        anime_id        : int
        name            : str
        content_score   : float (cosine similarity, 0.0-1.0)
    Sorted DESCENDING by content_score.
    """

    if not selected_titles:
        raise ValueError("Selected titles must contain at least one title.")
    
    #Step 1 - resolve titles to anime_ids
    selected_ids = _titles_to_ids(selected_titles, anime_df)
    if not selected_ids:
        raise ValueError(
            "None of the provided titles could be matched in the catalog."
        )
    
    #Step 2 - build centroid query vector
    centroid = _build_centroid(selected_ids, feature_matrix, anime_index_map)

    #Step 3 - compute cosine similarity between centroid & all anime
    similarity_scores = cosine_similarity(centroid, feature_matrix).flatten()

    #Step 4 - build results dataframe
    results = pd.DataFrame({
        "anime_id": anime_df["anime_id"].values,
        "name": anime_df["name"].values,
        "content_score": similarity_scores
    })

    #Step 5 - exclude the user's input titles from results!
    results = results[~results["anime_id"].isin(selected_ids)]

    #Step 6 - sort by similarity DESC & return top_n candidates
    results = results.sort_values("content_score", ascending=False).head(top_n).reset_index(drop=True)

    return results