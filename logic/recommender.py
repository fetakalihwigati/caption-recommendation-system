# # logic/recommender.py
# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score

# # =====================
# # LOAD MODEL & DATA
# # =====================

# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")
#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.npz")

#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================

# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     brand: str | None = None,
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()

#     # Filter brand
#     if brand and brand != "Semua":
#         df = df[df["brand"] == brand]
#         tfidf_matrix = tfidf_matrix[df.index]

#     # Vectorize input
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     # Similarity
#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # Final score (HASIL GRID SEARCH)
#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [["caption", "brand", "platform", "final_score", "link"]]
#     )

# # logic/recommender.py
# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import load_npz

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score

# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")
#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================
# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     brand: str | None = None,
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()

#     if brand and brand != "Semua":
#         df = df[df["brand"] == brand]
#         tfidf_matrix = tfidf_matrix[df.index]

#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [["caption", "brand", "platform", "final_score", "link"]]
#     )


# Bisa tidak error
# logic/recommender.py

# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score


# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")
#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     # Validasi kolom quality_score
#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================
# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     brand: str | None = None,
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()

#     # ⛑️ COPY agar aman dari SettingWithCopy
#     df = df.copy()

#     # =====================
#     # FILTER BRAND
#     # =====================
#     if brand and brand != "Semua":
#         mask = df["brand"] == brand
#         df = df[mask]
#         tfidf_matrix = tfidf_matrix[mask.values]

#     # =====================
#     # SIMILARITY
#     # =====================
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # =====================
#     # FINAL SCORE
#     # =====================
#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [["caption", "brand", "platform", "final_score", "link"]]
#     )

# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score


# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")
#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # # =====================
# # # MAIN RECOMMENDER
# # # =====================
# # def recommend_caption(
# #     input_caption: str,
# #     top_k: int = 5,
# #     meta_theme: str = "Semua",
# #     w_sim: float = 0.85,
# #     w_qual: float = 0.15
# # ):
# #     df, vectorizer, tfidf_matrix = load_assets()
# #     df = df.copy()

# #     # =====================
# #     # FILTER META THEME
# #     # =====================
# #     if meta_theme != "Semua":
# #         mask = df["meta_theme"] == meta_theme
# #         df = df[mask]
# #         tfidf_matrix = tfidf_matrix[mask.values]

# #     # =====================
# #     # GUARD EMPTY DATA
# #     # =====================
# #     if df.empty:
# #         return pd.DataFrame(
# #             columns=[
# #                 "caption",
# #                 "brand",
# #                 "platform",
# #                 "meta_theme",
# #                 "final_score",
# #                 "link"
# #             ]
# #         )

# #     # =====================
# #     # SIMILARITY
# #     # =====================
# #     input_clean = clean_text(input_caption)
# #     input_vec = vectorizer.transform([input_clean])

# #     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
# #     df["similarity_score"] = sim_scores

# #     # =====================
# #     # FINAL SCORE
# #     # =====================
# #     df["final_score"] = (
# #         w_sim * df["similarity_score"] +
# #         w_qual * df["quality_score"]
# #     )

# #     return (
# #         df.sort_values("final_score", ascending=False)
# #           .head(top_k)
# #           [
# #               [
# #                   "caption",
# #                   "brand",
# #                   "platform",
# #                   "meta_theme",
# #                   "final_score",
# #                   "link"
# #               ]
# #           ]
# #     )


# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     meta_theme: str = "Semua",
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()
#     df = df.copy()

#     # =====================
#     # SIMILARITY (GLOBAL)
#     # =====================
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # =====================
#     # FINAL SCORE
#     # =====================
#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     # =====================
#     # FILTER META THEME (SETELAH SCORING)
#     # =====================
#     if meta_theme != "Semua":
#         df = df[df["meta_theme"] == meta_theme]

#     if df.empty:
#         return pd.DataFrame(
#             columns=[
#                 "caption",
#                 "brand",
#                 "platform",
#                 "meta_theme",
#                 "final_score",
#                 "link"
#             ]
#         )

#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [
#               [
#                   "caption",
#                   "brand",
#                   "platform",
#                   "meta_theme",
#                   "final_score",
#                   "link"
#               ]
#           ]
#     )

#terakhir
# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score

# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")

#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     # Pastikan quality_score ada
#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================
# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     meta_theme: str = "Semua",
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()
#     df = df.copy()

#     # =====================
#     # FILTER META THEME
#     # =====================
#     if meta_theme != "Semua":
#         mask = df["meta_theme"].str.lower() == meta_theme.lower()

#         df = df.loc[mask].reset_index(drop=True)
#         tfidf_matrix = tfidf_matrix[mask.values]

#     # =====================
#     # SAFETY CHECK
#     # =====================
#     if len(df) == 0:
#         return pd.DataFrame()

#     if tfidf_matrix.shape[0] != len(df):
#         raise ValueError(
#             f"Mismatch df ({len(df)}) vs tfidf_matrix ({tfidf_matrix.shape[0]})"
#         )

#     # =====================
#     # SIMILARITY
#     # =====================
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # =====================
#     # FINAL SCORE
#     # =====================
#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     # =====================
#     # OUTPUT
#     # =====================
#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [[
#               "caption",
#               "meta_theme",
#               "brand",
#               "platform",
#               "engagement",
#               "similarity_score",
#               "quality_score",
#               "final_score",
#               "link"
#           ]]
#     )


# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score


# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")
#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================
# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     brands: list[str] | None = None,
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()
#     df = df.copy()

#     # =====================
#     # FILTER BY BRAND
#     # =====================
#     if brands:
#         mask = df["brand"].isin(brands)

#         if mask.sum() == 0:
#             return pd.DataFrame()

#         df = df[mask]
#         tfidf_matrix = tfidf_matrix[mask.values]

#     # # =====================
#     # # FILTER BY Tema
#     # # =====================
#     # if meta_theme:
#     #     mask = df["meta_theme"].isin(meta_theme)

#     #     if mask.sum() == 0:
#     #         return pd.DataFrame()

#     #     df = df[mask]
#     #     tfidf_matrix = tfidf_matrix[mask.values]
 

#     # =====================
#     # SIMILARITY
#     # =====================
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # =====================
#     # FINAL SCORE
#     # =====================
#     df["final_score"] = (
#         w_sim * df["similarity_score"]
#         + w_qual * df["quality_score"]
#     )

#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [[
#               "caption",
#               "brand",
#               "platform",
#               "meta_theme",
#               "engagement",
#               "final_score",
#               "similarity_score",
#               "quality_score",
#               "link"
#           ]]
#     )

# terbaru 
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from logic.preprocessing import clean_text
from logic.scoring import compute_quality_score


# =====================
# LOAD MODEL & DATA
# =====================
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_assets():
    df = pd.read_parquet("data/captions_final.parquet")

    # FIT TF-IDF SAAT RUNTIME
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["caption_clean"])

    df = compute_quality_score(df)

    return df, vectorizer, tfidf_matrix

#@st.cache_resource
#def load_assets():
 #   df = pd.read_parquet("data/captions_final.parquet")
#
 #   vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
  #  tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

    # pastikan kolom quality_score tersedia
   # df = compute_quality_score(df)

    #return df, vectorizer, tfidf_matrix


# =====================
# MAIN RECOMMENDER
# =====================
def recommend_caption(
    input_caption: str,
    top_k: int = 5,
    brand: str = "Semua",
    platform: str = "Semua",
    w_sim: float = 0.85,
    w_qual: float = 0.15
):
    df, vectorizer, tfidf_matrix = load_assets()
    df = df.copy()

    # =====================
    # FILTER BRAND
    # =====================
    if brand != "Semua":
        mask = df["brand"].str.lower() == brand.lower()
        df = df.loc[mask].reset_index(drop=True)
        tfidf_matrix = tfidf_matrix[mask.values]

    # =====================
    # FILTER PLATFORM
    # =====================
    if platform != "Semua":
        mask = df["platform"].str.upper() == platform.upper()
        df = df.loc[mask].reset_index(drop=True)
        tfidf_matrix = tfidf_matrix[mask.values]

    # =====================
    # SAFETY CHECK
    # =====================
    if len(df) == 0:
        return pd.DataFrame()

    if tfidf_matrix.shape[0] != len(df):
        raise ValueError(
            f"Mismatch df ({len(df)}) vs tfidf_matrix ({tfidf_matrix.shape[0]})"
        )

    # =====================
    # SIMILARITY
    # =====================
    input_clean = clean_text(input_caption)
    input_vec = vectorizer.transform([input_clean])

    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    df["similarity_score"] = sim_scores

    # =====================
    # FINAL SCORE
    # =====================
    df["final_score"] = (
        w_sim * df["similarity_score"] +
        w_qual * df["quality_score"]
    )

    # =====================
    # OUTPUT
    # =====================
    return (
        df.sort_values("final_score", ascending=False)
          .head(top_k)
          [[
              "caption",
              "brand",
              "platform",
              "engagement",
              "final_score",
              "similarity_score",
              "quality_score",
              "link"
          ]]
    )

# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from logic.preprocessing import clean_text
# from logic.scoring import compute_quality_score


# # =====================
# # LOAD MODEL & DATA
# # =====================
# @st.cache_resource
# def load_assets():
#     df = pd.read_parquet("data/captions_final.parquet")

#     vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
#     tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

#     # Pastikan quality_score tersedia
#     df = compute_quality_score(df)

#     return df, vectorizer, tfidf_matrix


# # =====================
# # MAIN RECOMMENDER
# # =====================
# def recommend_caption(
#     input_caption: str,
#     top_k: int = 5,
#     brand: str = "Semua",
#     platform: str = "Semua",
#     w_sim: float = 0.85,
#     w_qual: float = 0.15
# ):
#     df, vectorizer, tfidf_matrix = load_assets()
#     df = df.copy()

#     # =====================
#     # FILTER BRAND
#     # =====================
#     if brand != "Semua":
#         mask_brand = df["brand"].str.lower() == brand.lower()
#     else:
#         mask_brand = pd.Series([True] * len(df))

#     # =====================
#     # FILTER PLATFORM
#     # =====================
#     if platform != "Semua":
#         mask_platform = df["platform"].str.upper() == platform.upper()
#     else:
#         mask_platform = pd.Series([True] * len(df))

#     # =====================
#     # APPLY FILTER
#     # =====================
#     mask = mask_brand & mask_platform
#     df = df.loc[mask].reset_index(drop=True)
#     tfidf_matrix = tfidf_matrix[mask.values]

#     # =====================
#     # SAFETY CHECK
#     # =====================
#     if len(df) == 0:
#         return pd.DataFrame()

#     if tfidf_matrix.shape[0] != len(df):
#         raise ValueError(
#             f"Mismatch df ({len(df)}) vs tfidf_matrix ({tfidf_matrix.shape[0]})"
#         )

#     # =====================
#     # SIMILARITY
#     # =====================
#     input_clean = clean_text(input_caption)
#     input_vec = vectorizer.transform([input_clean])

#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     df["similarity_score"] = sim_scores

#     # =====================
#     # FINAL SCORE
#     # =====================
#     df["final_score"] = (
#         w_sim * df["similarity_score"] +
#         w_qual * df["quality_score"]
#     )

#     # =====================
#     # OUTPUT
#     # =====================
#     return (
#         df.sort_values("final_score", ascending=False)
#           .head(top_k)
#           [[
#               "caption",
#               "brand",
#               "platform",
#               "engagement",
#               "similarity_score",
#               "quality_score",
#               "final_score",
#               "link"
#           ]]
#     )
