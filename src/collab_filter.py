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
        inner_id = svd_model.trainset.to_inner_iid(str(anime_id))
        return svd_model.qi[inner_id]
    except ValueError:
        #anime_id was not seen during SVD training (filtered out)
        print(f"WARNING: anime_id {anime_id} not in SVD trainset - skip ping.")
        return None
    
