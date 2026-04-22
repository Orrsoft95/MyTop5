"""
preprocess.py
------------
Loads raw Kaggle CSVs, cleans & transforms them,
and then serializes the outputs needed by the recommendation
engine to models/

Outputs
------------
anime_metadata.pkl          : cleaned anime dataframe
content_feature_matrix.pkl  : combined TF-IDF + genre matrix
anime_index_map.pkl         : dict mapping anime_id to matrix row index
svd_model.pkl               : trained SVD model
anime_titles.pkl            : sorted list of titles for use in streamlit dropdown (select your top 5)

Run
---
python src/preprocess.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import SVD, Dataset, Reader