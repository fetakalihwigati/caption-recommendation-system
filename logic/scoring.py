# # logic/scoring.py
# import numpy as np

# def normalize(series):
#     return (series - series.min()) / (series.max() - series.min() + 1e-9)

# def compute_quality_score(df):
#     df = df.copy()

#     df["engagement_norm"] = normalize(df["engagement"])
#     df["quality_score"] = df["engagement_norm"]

#     return df

# logic/scoring.py

# def compute_quality_score(df):
#     """
#     Data sudah memiliki kolom quality_score dari notebook.
#     Fungsi ini hanya memastikan kolom tersedia.
#     """
#     required_cols = ["quality_score"]

#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Kolom '{col}' tidak ditemukan di data")

#     return df

# logic/scoring.py

import pandas as pd
import numpy as np


def normalize_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize engagement score using max normalization.

    engagement_norm = engagement / max(engagement)

    This follows the final research architecture:
    - engagement is used as performance signal
    - combined with similarity via multiplicative ranking
    """

    if "engagement" not in df.columns:
        raise ValueError("Kolom 'engagement' tidak ditemukan di dataset")

    df = df.copy()

    max_engagement = df["engagement"].max()

    # Safety check (avoid division by zero)
    if max_engagement == 0 or pd.isna(max_engagement):
        df["engagement_norm"] = 0.0
    else:
        df["engagement_norm"] = df["engagement"] / max_engagement

    return df

