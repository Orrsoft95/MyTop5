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

MAL_FIELDS = "mean,main_picture,genres"
TIMEOUT_LENGTH = 15

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


