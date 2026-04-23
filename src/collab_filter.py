"""
collab_filter.py
---------------
Collaborative filtering for the anime recommendation engine.

Given a list of anime_ids selected by the user, constructs a pseudo-user profile
from the SVD model's learned item factors & geenrates predicted ratings across
the full catalog.

This approach handles the cold-start problem; since the user has no rating history
in the training data, we approximate their latent factor vector by averaging
the item factor vectors of their selected titles.

Inputs (from models/)
--------------------
svd_model.pkl       : trained Surprise SVD model
anime_metadata.pkl  : cleaned anime dataframe

Public API
----------
    get_collab_recommendations(
        selected_titles : list[str],
        anime_df        : pd.DataFrame,
        svd_model       : surprise.SVD,
        top_n           : int=50
    ) -> pd.DataFrame
"""

import numpy as np
import pandas as pd
from surprise import SVD
from src.content_filter import _titles_to_ids

#Internal helper functions!

def _get_item_factor(svd_model: SVD, anime_id: int) -> np.ndarray | None:
    """
    Retrieve the latent item factor vector for a given anime_id from the
    trained SVD model.

    Surprise stores item factors in svd_model.qi, indexed by the model's internal item index.
    The trainset's to_inner_iid() method handles the mapping.

    Parameters
    ------------
    svd_model   : trained Surprise SVD model
    anime_id    : raw MAL anime_id integer

    Returns
    -------
    np.ndarray of shape (n_factors,) if found, else None
    """

    try:
        inner_id = svd_model.trainset.to_inner_iid(int(anime_id))
        return svd_model.qi[inner_id]
    except ValueError:
        #anime_id was not seen during SVD training (filtered out)
        print(f"WARNING: anime_id {anime_id} not in SVD trainset - skip ping.")
        return None
    
def _build_pseudo_user_vector(
        selected_ids: list[int],
        svd_model: SVD,
) -> np.ndarray:
    """
    Build a pseudo-user latent factor vector by averaging the item factor vectors
    of the user's selected anime.

    This approximates where the user would sit in the latent factor space learned
    by SVD, enabling rating prediction without any training history.

    Parameters
    ------------
    selected_ids    : list of anime_id integers
    svd_model       : trained Surprise SVD model

    Returns
    -------
    pseudo_user_vector  : np.ndarray of shape (n_factors,)
    """

    item_factors = []
    for anime_id in selected_ids:
        factor = _get_item_factor(svd_model, anime_id)
        if factor is not None:
            item_factors.append(factor)

    if not item_factors:
        raise ValueError(
            "None of the selected anime were found in the SVD trainset."
            "This may mean they were filtered out during preprocessing due to insufficient ratings."
            "Try selecting more popular titles."
        )
    
    return np.mean(item_factors, axis=0)

def _predict_ratings(
        pseudo_user_vector: np.ndarray,
        svd_model: SVD,
        exclude_ids: set,
        anime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict ratings for ALL catalog anime using the pseudo-user vector.

    For each anime in the catalog, the predicted rating is computed as:
        rating = global_mean + user_bias + item_bias + (pseudo_user * item_factor)

    This mirrors Surprise's internal SVD prediction formula, but substitutes
    the pseudo-user vector in place of a trained user factor vector.

    Parameters
    ----------
    pseudo_user_vector  : np.ndarray of shape (n_factors,)
    svd_model           : trained Surprise SVD model
    exclude_ids         : set of anime_ids to EXCLUDE (user's input titles)
    anime_df            : cleaned anime metadata DataFrame

    Returns
    -------
    pd.DataFrame with columns:
        anime_id, name_collab_score
    """
    trainset = svd_model.trainset
    global_mean = trainset.global_mean
    results = []

    for anime_id, name in zip(anime_df["anime_id"], anime_df["name"]):
        #Make sure we skip the input titles!
        if anime_id in exclude_ids:
            continue
        
        try:
            inner_id = trainset.to_inner_iid(int(anime_id))
            item_bias = svd_model.bi[inner_id]
            item_factor = svd_model.qi[inner_id]
        except ValueError:
            #If id not in trainset, just silently skip it
            continue

        #Create the SVD prediction formula (since this is a pseudo-user, there's no user bias)
        
        predicted_rating = (
            global_mean
            + item_bias
            + np.dot(pseudo_user_vector, item_factor)
        )

        results.append({
            "anime_id": anime_id,
            "name": name,
            "collab_score": predicted_rating
        })
    
    return pd.DataFrame(results)

# Public API
def get_collab_recommendations(
        selected_titles: list[str],
        anime_df: pd.DataFrame,
        svd_model: SVD,
        top_n: int=50,
) -> pd.DataFrame:
    """
    Return the top_n highest predicted-rating anime for a given list of titles
    using collaborative filtering via a pseudo-user SVD approach.

    Parameters
    ----------
    selected_titles : List of anime title strings (from Streamlit dropdown)
    anime_df        : cleaned anime metadata DataFrame
    svd_model       : Trained Surprise SVD model
    top_n           : Number of candidates to return (default 50, will be narrowed further
                        by hybrid.py)
    
    Returns
    -------
    pd.DataFrame with columns:
        anime_id        : int
        name            : str
        collab_score    : float (predicted rating on the model's rating scale)
    Sorted DESCENDING by collab_score.
    """

    if not selected_titles:
        raise ValueError("selected_titles must contain at least one title.")
    
    #Step 1 - resolve titles to their anime_ids
    selected_ids = _titles_to_ids(selected_titles, anime_df)
    if not selected_ids:
        raise ValueError(
                        "None of the provided titles could be matched in the catalog."
        )
    
    #Step 2 - build pseudo-user latent factor vector
    pseudo_user_vector = _build_pseudo_user_vector(selected_ids, svd_model)

    #Step 3 - Predict ratings across the full catalog
    results = _predict_ratings(
        pseudo_user_vector=pseudo_user_vector,
        svd_model=svd_model,
        exclude_ids=set(selected_ids),
        anime_df=anime_df,
    )

    if results.empty:
        raise ValueError(
            "No collaborative filtering predictions could be generated."
            "Check that the SVD model was trained successfully."
        )
    
    #Step 4 - sort by predicted rating DESC & return top_n candidates
    results = (
        results
        .sort_values("collab_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return results