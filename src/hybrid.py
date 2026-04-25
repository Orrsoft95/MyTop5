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
        collab_weight       : float = 0.4,
        exclude_ids         : set = None
    ) -> pd.DataFrame
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD

from src.content_filter import get_content_recommendations
from src.collab_filter import get_collab_recommendations

from src.mal_api import get_related_anime_ids

#Set default weights
CONTENT_WEIGHT = 0.55
COLLAB_WEIGHT = 0.45

#Set the # of candidates to request from each filter pre-merge
#Larger pool = more candidates for the hybrid merger to re-rank!
CANDIDATE_POOL = 300

#Internal helper functions!

def _normalize_scores(df: pd.DataFrame, column:str) -> pd.DataFrame:
    """
    Apply Min-Max normalization to a score column, ensuring values are set to [0,1].

    If all values are identical (no variance), return 0.5 for ALL rows instead of NaN to avoid
    breaking the hybrid model.

    Parameters
    ----------
    df      : DataFrame containing the score column
    column  : name of the column to normalize (in place)

    Returns
    -------
    DataFrame with the specified column normalized to [0,1]
    """

    values = df[column].values.reshape(-1, 1)

    if values.max() == values.min():
        #This is our 0-variance edge case; set all scores to 0.5
        df[column] = 0.5
        return df
    
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(values)
    return df

def _filter_related_titles(
        results         : pd.DataFrame,
        selected_titles : list[str],
        top_n_exempt    : int=1,
) -> pd.DataFrame:
    """
    REMOVE recommendations whose titles contain (or are contained by)
    any of the user's selected titles, UNLESS they rank in  the top_n_exempt.

    For example, if "Fullmetal Alchemist" is suggested, "Fullmetal Alchemist: Brotherhood"
    will be filtered out unless it's in the top 1 results.

    Parameters
    ----------
    results         : ranked recommendations DataFrame with "name" column
    selected_titles : list of user-selected title strings
    top_n_exempt    : top N results are immune to filtering (default, 1)

    Returns
    -------
    filtered DataFrame with related titles removed outside the exempt zone
    """

    selected_lower = [t.strip().lower() for t in selected_titles]

    def is_related(title:str) -> bool:
        #ignore these words when comparing titles - they're too common
        stopwords = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "at"}

        #ONLY use the root segment of the recommended title (before : or -)
        root_title = title.strip().lower().split(":")[0].split(" - ")[0].strip()
        t_tokens = set(root_title.split()) - stopwords
        t_full = title.strip().lower()
        
        for s in selected_lower:
            #Use only the root segment of selected title as well
            root_s = s.split(":")[0].split(" - ")[0].strip()
            s_tokens = set(root_s.split()) - stopwords

            #Check #1: ensure recommended anime title is not IN selected
            #anime title (or vice-versa)
            if t_full in s or s in t_full:
                return True

            #Percentage overlap check using Jaccard similarity!
            #(intersection / union of the 2 token sets)
            #If recommended anime shares >=50% of its ROOT title
            #w/ a top 5 anime, exclude it.
            if not t_tokens or not s_tokens:
                continue

            overlap = t_tokens & s_tokens
            union = t_tokens | s_tokens
            similarity = len(overlap) / len(union)

            if similarity >= 0.5:
                return True
        return False
    
    #Split into exempt (top 3) and filterable (the rest)
    exempt = results.iloc[:top_n_exempt]
    filterable = results.iloc[top_n_exempt:]

    #Filter related titles ONLY from the non-exempt portion
    filterable = filterable[~filterable["name"].apply(is_related)]

    return pd.concat([exempt, filterable], ignore_index=True)

def _greedy_deduplicate(
        results         : pd.DataFrame,
        selected_titles : list[str],
        anime_df        : pd.DataFrame,
        client_id       : str,
        exclude_ids     : set = None,
        top_n           : int = 10,
) -> pd.DataFrame:
    """
    Greedily re-rank results so that NO recommendation is name-similar
    to any other OR MAL-related to any other already-selected recommendation.
    """
    if exclude_ids is None:
        exclude_ids = set()
    
    running_exclude_ids = set(exclude_ids)
    running_titles = list(selected_titles)
    final = []

    for _, row in results.iterrows():
        candidate_id = int(row["anime_id"])
        candidate_title = row["name"]

        #Check 1 - MAL-related exclusion set
        if candidate_id in running_exclude_ids:
            continue

        #Check 2 - name similarity against all CURRENTLY selected
        #titles.
        candidate_df = pd.DataFrame([row])
        filtered = _filter_related_titles(
            candidate_df,
            running_titles,
            top_n_exempt=0 #Make NO exemptions here
        )

        if filtered.empty:
            continue

        #Both checks are passed at this point - add to final list!
        final.append(row)
        running_exclude_ids.add(candidate_id)
        running_titles.append(candidate_title)

        #Fetch MAL-related IDs for this recommendation & add to
        #running_exlcude_ids so future candidates are checked against it!
        related = get_related_anime_ids(
            selected_titles=[candidate_title],
            anime_df=anime_df,
            client_id=client_id
        )
        running_exclude_ids.update(related)

        if len(final) == top_n:
            break

    return pd.DataFrame(final).reset_index(drop=True)

#Public API

def get_hybrid_recommendations(
    selected_titles     : list[str],
    anime_df            : pd.DataFrame,
    feature_matrix      : csr_matrix,
    anime_index_map     : dict,
    svd_model           : SVD,
    top_n               : int = 10,
    content_weight      : float = CONTENT_WEIGHT,
    collab_weight       : float = COLLAB_WEIGHT,
    exclude_ids         : set = None,
    client_id           : str = None,
) -> pd.DataFrame:
    """
    Generate HYBRID anime recommendations by fusing content-based and collaborative filtering scores.

    Pipeline
    --------
    1. Get top CANDIDATE_POOL content-based candidates (cosine similarity)
    2. Get top CANDIDATE_POOL collaborative candidates (SVD pseudo-user)
    3. Merge candidates on anime_id (OUTER JOIN - keep titles, even if they were only scored by one of the systems.)
    4. Normalize both score columns to [0,1]
    5. Compute weighted hybrid score
    6. Sort by hybrid score and return top_n results with metadata

    Parameters
    ----------
    selected_titles : list of anime title strings (from Streamlit dropdown)
    anime_df        : cleaned anime metadata DataFrame
    feature_matrix  : sparse TF-IDF + genre feature matrix
    anime_index_map : dict mapping anime_id to matrix row index
    svd_model       : trained Surprise SVD model
    top_n           : number of final recommendations to return (default 10)
    content_weight  : weight applied to normalized content score (default 0.6)
    collab_weight   : weight applied to normalized collab score (default 0.4)
    client_id       : MAL API client ID from st.secrets

    Returns
    -------
    pd.DataFrame with columns:
        anime_id        : int
        name            : str
        content_score   : float (normalized 0-1)
        collab_score    : float (normalized 0-1)
        hybrid_score    : float (weighted blend, 0-1)
        genres          : str
        synopsis        : str
    Sorted DESCENDING by hybrid_score, top_n rows.
    """

    if not selected_titles:
        raise ValueError("Selected_titles must contain at least one title.")
    
    if not np.isclose(content_weight + collab_weight, 1):
        raise ValueError(
            f"ERROR: content_weight ({content_weight} + collab_weight ({collab_weight}) must sum to 1.0!)"
        )
    
    #Step 1 - pull content-based candidates
    print("Generating content-based candidates...")
    content_df = get_content_recommendations(
        selected_titles=selected_titles,
        anime_df=anime_df,
        feature_matrix=feature_matrix,
        anime_index_map=anime_index_map,
        top_n=CANDIDATE_POOL
    )

    #Step 2 - pull collaborative candidates
    print("Now generating collaborative filtering candidates...")
    collab_df = get_collab_recommendations(
        selected_titles=selected_titles,
        anime_df=anime_df,
        svd_model=svd_model,
        top_n=CANDIDATE_POOL
    )

    #Step 3 - Merge (OUTER JOIN) candidate pools on anime_id
    merged_df = pd.merge(
        content_df[["anime_id", "content_score"]],
        collab_df[["anime_id", "collab_score"]],
        on="anime_id",
        how="outer"
    )

    #OUTER Joins keep anime even if they are Missing one of the 2 scores, impute these missing scores with that column's average.
    merged_df["content_score"] = merged_df["content_score"].fillna(merged_df["content_score"].mean())
    merged_df["collab_score"] = merged_df["collab_score"].fillna(merged_df["collab_score"].mean())

    #Make Sure the merge isn't empty
    if merged_df.empty:
        raise ValueError(
            "ERROR: No anime appeared in both content AND collaborative candidate pools. "
            "Try increasing CANDIDATE_POOL or selecting different input titles."
        )
    
    print(f"Both candidate pools have been generated! # of candidates in the merged pool: {len(merged_df)}")

    #Step 4 - normalize BOTH scores to [0,1]
    merged_df = _normalize_scores(merged_df, "content_score")
    merged_df = _normalize_scores(merged_df, "collab_score")

    #Step 5 - compute the weighted hybrid score
    merged_df["hybrid_score"] = (
        content_weight * merged_df["content_score"]
        + collab_weight * merged_df["collab_score"]
    )

    #Step 6 - sort by hybrid score
    merged_df = (
        merged_df
        .sort_values("hybrid_score", ascending=False)
        .reset_index(drop=True)
    )

    #Step 7 - enrich results w/ metadata from anime_df
    metadata_cols = ["anime_id", "name", "genres", "synopsis"]
    available_cols = [x for x in metadata_cols if x in anime_df.columns]
    metadata = anime_df[available_cols]

    results = pd.merge(
        merged_df,
        metadata,
        on="anime_id",
        how="left"
    )

    #Step 7b - exclude any MAL-defined related titles
    if exclude_ids:
        results = results[~results["anime_id"].isin(exclude_ids)]

    #Step 7c - Exclude any name-similar titles not caught by the MAL API
    results = _filter_related_titles(results, selected_titles, top_n_exempt=1)

    #Step 7d - Use greedy deduplication to remove recommendations that are related
    #to one another, and generate our final top 10 list!
    results = _greedy_deduplicate(
        results=results,
        selected_titles=selected_titles,
        anime_df=anime_df,
        client_id=client_id,
        exclude_ids=exclude_ids,
        top_n=top_n
    )

    #Set the final column order
    col_order = [
        "anime_id",
        "name",
        "content_score",
        "collab_score",
        "hybrid_score",
        "genres",
        "synopsis"
    ]

    col_order = [x for x in col_order if x in results.columns]
    results = results[col_order]

    print(f"All set! Returning the top {len(results)} hybrid recommendations now.")
    return results