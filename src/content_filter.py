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
