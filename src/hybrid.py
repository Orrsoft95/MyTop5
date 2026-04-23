"""
hybrid.py
----------
Hybridized score fusion for the anime recommendation engine.

Combines content-based similarity scores & collaborative filtering's
predicted ratings into a single ranked list of recommendations.

Both scores are normalized to [0,1] before blending to ensure neither method dominates
due to differences in scale (cosine similarity is already 0-1, but SVD predicted ratings
matches MAL's 1-10 rating scale).

The weighted blend is tunable via CONTENT_WEIGHT and COLLAB_WEIGHT.

Public API
----------
    get_hybrid_recommendations(
        selected_titles     : list[str],
        anime_df            : pd.DataFrame,
        feature_matrix      : scipy sparse matrix,
        anime_index_map     : dict,
        svd_model           : Surprise.SVD,
        top_n               : int = 10,
        content_weight      : float = 0.6,
        collab_weight       : float = 0.4
    ) -> pd.DataFrame
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD

from src.content_filter import get_content_recommendations
from src.collab_filter import get_collab_recommendations

#Set default weights
CONTENT_WEIGHT = 0.6
COLLAB_WEIGHT = 0.4

#Set the # of candidates to request from each filter pre-merge
#Larger pool = more candidates for the hybrid merger to re-rank!
CANDIDATE_POOL = 100

