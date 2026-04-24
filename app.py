"""
app.py
-------
Streamlit front-end for the MyTop5 anime recommendation engine.

Users select their top 5 favorite anime via a searchable autocomplete
dropdown and receive 10 personalized recommendations, displayed as enriched
result cards in a 2-column grid.

Models are loaded from Hugging Face Hub on first run & cached for the remainder
of the session via st.cache_resource.

Run locally
------------
    streamlit run app.py

Secrets required (.streamlit/secrets.toml)
------------------------------------------
    [mal]
    client_id = "your_client_id"

    [huggingface]
    repo_id = "hf_username/MyTop5"
"""

import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from streamlit_searchbox import st_searchbox

from src.hybrid import get_hybrid_recommendations
from src.mal_api import enrich_recommendations

#Page config
st.set_page_config(
    page_title="MyTop5 - Anime Recommender",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="collapsed"

)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #fafafa; }
 
    /* Result card */
    .anime-card {
        background-color: #1a1d27;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 100%;
    }
 
    /* Card title */
    .anime-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e8eaf6;
        margin-bottom: 0.4rem;
    }
 
    /* Metadata pill */
    .meta-pill {
        display: inline-block;
        background-color: #2e3250;
        color: #9fa8da;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 2px 2px 6px 0;
    }
 
    /* Score badge */
    .score-badge {
        display: inline-block;
        background-color: #3949ab;
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
 
    /* Synopsis text */
    .synopsis-text {
        font-size: 0.8rem;
        color: #9e9e9e;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
 
    /* Hybrid score bar label */
    .score-label {
        font-size: 0.75rem;
        color: #7986cb;
        margin-bottom: 2px;
    }
 
    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e8eaf6;
        margin: 1.5rem 0 1rem;
        border-bottom: 1px solid #2e3250;
        padding-bottom: 0.5rem;
    }
 
    /* MAL link */
    a.mal-link {
        color: #7986cb;
        font-size: 0.8rem;
        text-decoration: none;
    }
    a.mal-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

#Model Loading!
@st.cache_resource(show_spinner="Loading models from Hugging Face...")

def load_models():
    pass