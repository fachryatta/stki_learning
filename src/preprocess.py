import pandas as pd
import re


# ==========================
# Text Cleaning
# ==========================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================
# Load Food Dataset
# ==========================
def load_and_preprocess_food(path: str):
    df = pd.read_csv(path)

    if "Describe" not in df.columns:
        raise Exception("Kolom 'Describe' tidak ditemukan.")

    df["text"] = df["Describe"].apply(clean_text)
    return df


# ==========================
# Load Ratings Dataset
# ==========================
def load_and_clean_ratings(path: str):
    df = pd.read_csv(path)

    if "Food_ID" not in df.columns:
        raise Exception("Kolom 'Food_ID' tidak ditemukan di ratings.csv")

    if "Rating" in df.columns:
        df["Rating"] = df["Rating"].astype(float)

    return df
