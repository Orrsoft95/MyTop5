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
    """
    Download & deserialize all model files from Huggingface hub.
    Cached for the session - only runs once per deployment instance.
    """

    repo_id = st.secrets["huggingface"]["repo_id"]

    model_files = [
        "anime_metadata.pkl",
        "content_feature_matrix.pkl",
        "anime_index_map.pkl",
        "svd_model.pkl",
        "anime_titles.pkl"
    ]

    models = {}
    for filename in model_files:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        with open(path, "rb") as file:
            models[filename] = pickle.load(file)

    return(
        models["anime_metadata.pkl"],
        models["content_feature_matrix.pkl"],
        models["anime_index_map.pkl"],
        models["svd_model.pkl"],
        models["anime_titles.pkl"]
    )

def search_anime(query: str) -> list[str]:
    """
    Return anime titles matching the search query for the
    autocomplete dropdown. Called on every keystroke by st_searchbox.
    """

    if not query or len(query) < 2:
        return []
    query_lower = query.strip().lower()

    #Cap at 10 suggestions for performance
    return [
        title for title in st.session_state.anime_titles
        if query_lower in title.lower()
    ][:10]

def render_card(row: pd.Series) -> None:
    """
    Render a single recommendation result card using HTML/CSS.
    Displays cover art, title, MAL score, genres, synopsis, and hybrid score.
    """

    #Cover Art
    if pd.notna(row.get("cover_image_url")) and row["cover_image_url"]:
        st.image(row["cover_image_url"], width=200) #use_column_width=True)
    else:
        st.markdown("*(No cover art available)*")

    #TItle + MAL link
    mal_url = row.get("mal_url", "#")
    st.markdown(
        f'<div class="anime-title">'
        f'<a class="mal-link" href="{mal_url}" target="_blank">'
        f'{row["name"]}</a></div>',
        unsafe_allow_html=True
    )

    #MAL score badge
    mal_score = row.get("mal_score")
    if pd.notna(mal_score) and mal_score:
        st.markdown(
            f'<span class="score-badge">⭐ {mal_score:.2f} /10</span>',
            unsafe_allow_html=True
        )
    
    #Episode count
    num_episodes = row.get("num_episodes")
    if pd.notna(num_episodes) and num_episodes:
        st.markdown(
            f'<span class="meta-pill">📺  {int(num_episodes)} eps</span>',
            unsafe_allow_html=True
        )

    #Genres
    genres = row.get("genres")
    if pd.notna("genres") and genres:
        #MAL API returns genres as list of dicts, but
        #Is returned from our csv as a plain string - need to handle both cases
        if isinstance(genres, list):
            genre_names = sorted([g["name"] for g in genres if "name" in g])
        else:
            genre_names = [g.strip() for g in str(genres).split(",")]
        pills = "".join(
            f'<span class="meta-pill">{g}</span>' for g in genre_names[:5]
        )
        st.markdown(pills, unsafe_allow_html=True)

    #Synopsis - truncate to <=200 chars!
    synopsis = row.get("synopsis")
    if pd.notna(synopsis) and synopsis:
        truncated = str(synopsis)[:200] + "..." if len(str(synopsis)) > 200 else str(synopsis)
        st.markdown(
            f'<div class="synopsis-text">{truncated}</div>',
            unsafe_allow_html=True
        )

    #Hybrid score - progress bar
    hybrid_score = row.get("hybrid_score", 0)
    st.markdown(
        f'<div class="score-label">Match score: {hybrid_score:.0%}</div>',
        unsafe_allow_html=True
    )
    st.progress(float(hybrid_score))


### MAIN APP ###
def main():
    #Load models
    anime_df, feature_matrix, anime_index_map, svd_model, anime_titles = load_models()

    #Store titles in session state for the searchbox helper
    st.session_state.anime_titles = anime_titles

    #Header
    st.markdown("# 🏯 MyTop5")
    st.markdown(
        "Select your **top 5 favorite anime** and receive 10 personalized "
        "recommendations, powered by a hybrid content + collaborative filtering engine.\n"
        "**Note** - anime rekeased after 2023 may not be present in data set."
    )
    st.divider()

    #Anime selection!
    st.markdown("### 🔍 Select your top 5 anime")
    st.caption("Search by title - type at least 2 characters to see suggestions.")

    selected_titles = []
    for i in range(1, 6):
        title = st.selectbox(
            label=f"Pick #{i}",
            options=[""] + anime_titles,
            key=f"select_{i}",
            index=0,
            placeholder=f"Anime #{i}...",
        )
        if title:
            selected_titles.append(title)
    
    #Title validation
    if selected_titles:
        unique_titles = list(dict.fromkeys(selected_titles)) #Preserve entry order + deduplicate
        if len(unique_titles) < len(selected_titles):
            st.warning("You've selected the same anime more than once! Duplicates will be ignored.")
            selected_titles = unique_titles
    
    #"Get Recommendations" button
    st.divider()
    ready = len(selected_titles) == 5
    if not ready and selected_titles:
        st.info(f"{len(selected_titles)}/5 anime selected - pick {5 - len(selected_titles)} more to continue.")

    if st.button(
        "✨ Get Recommendations",
        disabled=not ready,
        use_container_width=True,
        type="primary",
    ):
        with st.status("Generating your recommendations...", expanded=True) as status:
            try:
                #Step 1 - hybrid recommendations
                status.update(label="Running hybrid recommendation engine...")
                results = get_hybrid_recommendations(
                    selected_titles=selected_titles,
                    anime_df=anime_df,
                    feature_matrix=feature_matrix,
                    anime_index_map=anime_index_map,
                    svd_model=svd_model,
                    top_n=10
                )
                
                #Step 2 - enrich w/ MAL API data
                status.update(label="Fetching cover art & scores from MyAnimeList...")
                client_id = st.secrets["mal"]["client_id"]
                
                results = enrich_recommendations(results, client_id)
                
                status.update(label="Done!", state="complete",expanded=False)

                # Store in session state so results persist after rerun
                st.session_state.results = results
                st.session_state.selected_titles = selected_titles


            except ValueError as e:
                status.update(label="Something went wrong.", state="error", expanded=False)
                st.error(f"Something went wrong while generating your recommendations: {e}")
            except Exception as e:
                status.update(label="An unexpected error occurred.", state="error", expanded=False)
                st.error(f"An unexpected error occurred: {e}")
                raise

    #Step 3 - display results (if they exist in session state)!
    if "results" in st.session_state and st.session_state.results is not None:
        results = st.session_state.results
        selected_titles_display = st.session_state.get("selected_titles", [])

        st.markdown(
            '<div class="section-header">💡 Your recommendations</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"Based on: {', '.join(selected_titles)}"
        )

        #Create our 2-column result grid
        rows = [results.iloc[i:i+2] for i in range(0, len(results), 2)]
        for row_pair in rows:
            cols = st.columns(2, gap="medium")
            for col, (_, anime_row) in zip(cols, row_pair.iterrows()):
                with col:
                    with st.container():
                        st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                        render_card(anime_row)
                        st.markdown('</div>',unsafe_allow_html=True)
    
    #Footer
    st.divider()
    st.caption(
        "Data sourced from the "
        "[Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) "
        "on Kaggle. Live metadata pulled from the "
        "[MyAnimeList API](https://myanimelist.net/apiconfig/references/api/v2)."
    )

if __name__ == "__main__":
    main()