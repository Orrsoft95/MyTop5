
# **🎌MyTop5 - Anime Recommendation Engine**

A hybrid anime recommendation system built as a Data Analytics Capstone project. Users select their top 5 favorite anime and receive 
10 personalized recommendations, powered by a combnination of content-based filtering & collaborative filtering.

**Live app:** [mytop5.streamlit.app](https://mytop5.streamlit.app)

---
title: MyTop5
page icon: 🎌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

----

## How It Works

MyTop5 uses a hybrid recommendation engine that combines two approaches:

- **Content-based filtering** -- matches anime by shared attributes (genre tags & synopsis text) using *TF-IDF vectorization* and *cosine similarity*.
-  **Collaborative Filtering** -- identifies anime that users with similar taste have rated highly, using *Singular Value Decomposition (SVD)* on a matrix of 20M+ MyAnimeList user ratings.
- **Hybrid fusion** -- combines both scores via weighted normalization (55% content, 45% collaborative) to balance relevance & novelty.

Result cards are enriched with live cover art, scores, and metadata from the [MyAnimeList API v2](https://myanimelist.net/apiconfig/references/api/v2).

---

## Project Structure
```
MyTop5/
├── data/                       	# Raw CSVs (not tracked by git — see Setup)
│   ├── anime-filtered.csv
│   └── users-score-2023.csv
├── models/                     	# Serialized model files (not tracked by git)
├── notebooks/
│   ├── 01_eda.ipynb            	# Exploratory data analysis & preprossessing
│   ├── 02_filtering.ipynb			# Testing content & collaborative filtering systems
│   ├── 03_API.ipynb				# Testing the MAL API. (not tracked by git - see Setup)
│   ├── 04_hybrid_vs_content.ipynb  # Hypothesis-testing experiments
├── src/
│   ├── __init__.py
│   ├── preprocess.py           	# Data cleaning, feature engineering, SVD training
│   ├── content_filter.py       	# Cosine similarity recommendation logic
│   ├── collab_filter.py        	# SVD pseudo-user recommendation logic
│   ├── hybrid.py               	# Score fusion and final ranking
│   ├── mal_api.py              	# MyAnimeList API client
│   └── upload_models.py        	# Uploads trained models to Hugging Face Hub
├── .streamlit/
│   ├── config.toml             	# Dark theme configuration
│   └── secrets.toml            	# API credentials (not tracked by git)
├── app.py                      	# Streamlit application entry point
├── environment.yml             	# Conda environment definition
├── LICENSE							# Copy of the MIT software license
├── MODEL_CARD.md 
└── README.md
```
---

## Setup & Local Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/Orrsoft95/MyTop5.git
cd MyTop5
```
### 2. Create & activate the conda environment
```bash
conda env create -f environment.yml
conda activate MyTop5
```
### 3. Download the data
Download the following files from the [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) on Kaggle, and place them in the `data/` folder:
- `anime-filtered.csv`
- `users-score-2023.csv`

### 4. Configure API credentials

Create `.streamlit/secrets.toml` with the following:
```toml
[mal]
client_id = "your_mal_client_id_here"
[huggingface]
repo_id = "your-hf-username/MyTop5"
```

To obtain a MAL API client ID, register an application at [myanimelist.net/apiconfig](https://myanimelist.net/apiconfig).

### 5. Run the preprocessing pipeline

```bash
python src/preprocess.py
```
This step cleans the raw CSVs, builds the content feature matrix, trains the SVD model, and saves all outputs to `models/`.
**NOTE:** this step can take *up to 20 minutes* to complete due to SVD hyperparameter tuning.

### 6. (Optional) Upload models to Hugging Face Hub
If you want to deploy to Streamlit Cloud, upload the trained models to Hugging Face:
```bash
export HF_TOKEN="your_hf_token_here"
python src/upload_models.py
```

### 7. Run the app locally
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

---
## Data Sources

| Dataset | Source | Description |
|---|---|---|
| `anime-filtered.csv` | [Anime Dataset 2023 — Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) | Anime metadata — titles, genres, synopsis, scores |
| `users-score-2023.csv` | [Anime Dataset 2023 — Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) | 20M+ user ratings from MyAnimeList |
| Live metadata | [MyAnimeList API v2](https://myanimelist.net/apiconfig/references/api/v2) | Cover art, current scores, MAL page links, episode counts |

---
## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data processing | pandas, numpy, scipy |
| Machine learning | scikit-learn, scikit-surprise |
| NLP | TF-IDF (scikit-learn) |
| App framework | Streamlit |
| Model hosting | Hugging Face Hub |
| API client | requests |

---
## Limitations

- Dataset covers anime through 2023 — titles released *after* that date are not in the catalog.
- MAL user ratings skew toward dedicated fans and may not reflect casual viewer preferences.
- Users with fewer than **50** ratings were excluded from SVD training to reduce noise, which may affect recommendations for niche titles.
- The pseudo-user cold-start approach approximates collaborative filtering for new users and does not have access to a trained user bias term.

---

## Acknowledgements

- [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) by dbdmobile on Kaggle
- [MyAnimeList](https://myanimelist.net) for the public API
- [Surprise](https://surpriselib.com) for the SVD collaborative filtering implementation