"""
mal_api.py
----------
MyAnimeList API v2 client for the anime recommendation engine.

Used exclusively at the UI layer to enrich recommendation result cards with cover art, MAL page URLs,
and current community scores. Not involved in model training or scoring.

API Documentation: https://myanimelist.net/apiconfig/references/api/v2

Authentication
--------------
Requires a MAL client ID stored in .streamlit/secrets.toml:
    [mal]
    client_id = "your_client_id"

Accessed in code via st.secrets["mal"]["client_id"]

Public API
----------
    enrich_recommendations(
        results_df  : pd.DataFrame,
        client_id   : str,
    ) -> pd.DataFrame
"""

import time
import requests
import pandas as pd


MAL_API_BASE   = "https://api.myanimelist.net/v2"
MAL_ANIME_URL  = "https://myanimelist.net/anime" #This link itself leads to a 404 page, but will work once anime_id is appended

"""
Fields to request from MAL API, per anime
    mean            : current community score
    main_picture    : cover art (med and large URLs)
    genres          : genre tags directly from MAL
"""

MAL_FIELDS = "mean,main_picture,genres,num_episodes"
TIMEOUT_LENGTH = 15
REQUEST_DELAY = 0.25

#Internal helper functions!

def _get_anime_details(anime_id: int, client_id: str) -> dict | None:
    """
    Fetch details for a single anime from the MAL API.

    Parameters
    ----------
    anime_id    : MAL anime_id integer
    client_id   : MAL API client ID from st.secrets

    Returns
    -------
    dict of API response fields, or None if request fails
    """

    url = f"{MAL_API_BASE}/anime/{anime_id}"
    headers = {"X-MAL-CLIENT-ID": client_id}
    params = {"fields":MAL_FIELDS}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_LENGTH)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for anime_id {anime_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for anime_id {anime_id}: {e}")
        return None

def _parse_anime_details(anime_id: int, data:dict) -> dict:
    """
    Extract & flatten the fields we need from the MAL API response.

    Handle missing fields by substituting sensible defaults so that a single
    failed field doesn't break the entire result card.

    Parameters
    ----------
    anime_id    : MAL anime_id integer
    data        : raw API response dict from _get_anime_details

    Returns
    ---------
    dict with keys: anime_id, mal_score, cover_image_url, mal_url, num_episodes
    """

    #Prefer the med size cover art, use large if needed, then none if both are missing
    main_picture = data.get("main_picture", {})
    cover_image_url = (
        main_picture.get("medium")
        or main_picture.get("large")
        or None
    )

    return {
        "anime_id": anime_id,
        "mal_score": data.get("mean", None),
        "cover_image_url": cover_image_url,
        "mal_url": f"{MAL_ANIME_URL}/{anime_id}",
        "num_episodes": data.get("num_episodes", None),
    }

# Public API

def enrich_recommendations(
        results_df: pd.DataFrame,
        client_id: str,
) -> pd.DataFrame:
    """
    Enrich a recommendations dataframe with live data from the MAL API.

    Makes one API call per recommended anime to fetch cover art, current community score, and MAL page URL.
    Results are joined back into the input dataframe by anime_id.

    A short delay (REQUEST_DELAY seconds) is applied between calls to respect MAL rate limits.

    Parameters
    ----------
    results_df  : DataFrame output from hybrid.get_hybrid_recommendations()
                    MUST contain an anime_id column.
    client_id   : MAL API client ID from st.secrets["mal"]["client_id"]

    Returns
    -------
    pd.DataFrame - original resuilts_df with 3 additional columns:
        mal_score       : float | None (current MAL community score)
        cover_image_url : str | None (URL for the title's cover art image)
        mal_url         : str (Link to the title's MAL page)
    """

    if results_df.empty:
        return results_df
    
    if "anime_id" not in results_df.columns:
        raise ValueError("ERROR: results_df must contain an 'anime_id' column.")
    
    if not client_id:
        raise ValueError(
            "MAL client ID is missing! "
            "Check that .streamlit/secrets.toml contains [mal] client_id."
        )
    
    enriched_rows = []
    total = len(results_df)

    for i, anime_id in enumerate(results_df["anime_id"], start=1):
        print(f"Fetching MAL data {i}/{total} - anime_id: {anime_id}...")

        data = _get_anime_details(int(anime_id), client_id)

        if data:
            enriched_rows.append(
                _parse_anime_details(int(anime_id), data)
                )
        else:
            #If the API call failed, append placeholder row so that merge still works.
            enriched_rows.append({
                "anime_id": int(anime_id),
                "mal_score": None,
                "cover_image_url": None,
                "mal_url": f"{MAL_ANIME_URL}/{anime_id}"
            })

        #Add a sleep period to respect MAL API rate limits
        if i < total:
            time.sleep(REQUEST_DELAY)
        
    enriched_df = pd.DataFrame(enriched_rows)

    #join enrichment data back into results
    results_enriched = pd.merge(
        results_df,
        enriched_df,
        on="anime_id",
        how="left"
    )

    return results_enriched