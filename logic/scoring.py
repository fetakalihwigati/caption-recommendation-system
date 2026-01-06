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

def compute_quality_score(df):
    """
    Data sudah memiliki kolom quality_score dari notebook.
    Fungsi ini hanya memastikan kolom tersedia.
    """
    required_cols = ["quality_score"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di data")

    return df
