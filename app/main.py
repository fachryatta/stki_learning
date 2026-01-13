import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Rekomendasi Makanan",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    food_df = pd.read_csv("data/1662574418893344.csv")
    ratings_df = pd.read_csv("data/ratings.csv")

    if "Food_ID" in food_df.columns and "Food_ID" in ratings_df.columns:
        df = pd.merge(food_df, ratings_df, on="Food_ID", how="inner")
        return df

    st.error("Kolom Food_ID tidak ditemukan.")
    st.stop()


df = load_data()


# =========================================================
# TEXT PREPROCESSING
# =========================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_desc"] = df["Describe"].apply(clean_text)


# =========================================================
# TF-IDF
# =========================================================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_desc"])


# =========================================================
# IMAGE FROM BING
# =========================================================
@st.cache_data(show_spinner=False)
def get_food_image(food_name):
    if not isinstance(food_name, str) or food_name.strip() == "":
        return "https://via.placeholder.com/400x300?text=No+Image"

    query = food_name.lower().strip().replace(" ", "+")
    return f"https://tse1.mm.bing.net/th?q={query}+food"


# =========================================================
# UI HEADER
# =========================================================
st.markdown("""
<h1 style="text-align:center;">🍽️ Aplikasi Rekomendasi Makanan Luar Negeri</h1>
<p style="text-align:center;">Sistem rekomendasi berbasis TF-IDF & Cosine Similarity</p>
<hr>
""", unsafe_allow_html=True)


# =========================================================
# FILTER
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    kategori = st.selectbox(
        "Jenis Makanan",
        sorted(df["C_Type"].dropna().unique())
    )

with col2:
    tipe = st.selectbox(
        "Vegetarian / Non-Veg",
        sorted(df["Veg_Non"].dropna().unique())
    )

with col3:
    min_rating = st.slider("Rating Minimal", 0, 10, 7)


# =========================================================
# BUTTON
# =========================================================
if st.button("🔍 Carikan Rekomendasi"):

    filtered_df = df[
        (df["C_Type"] == kategori) &
        (df["Veg_Non"] == tipe) &
        (df["Rating"] >= min_rating)
    ]

    if filtered_df.empty:
        st.warning("Tidak ditemukan data sesuai filter.")
        st.stop()

    # ======================
    # CONTENT SIMILARITY
    # ======================
    query_text = " ".join(filtered_df["clean_desc"].tolist())
    query_vec = vectorizer.transform([query_text])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df["similarity"] = similarity_scores
    ranked = df.sort_values(
        by=["similarity", "Rating"],
        ascending=False
    ).drop_duplicates("Food_ID")

    main_rec = ranked.iloc[0]
    others = ranked.iloc[1:4]


    # =====================================================
    # MAIN RECOMMENDATION
    # =====================================================
    st.markdown("## ⭐ Rekomendasi Utama")

    col_img, col_text = st.columns([1, 2])

    with col_img:
        try:
            st.image(
                get_food_image(main_rec["Name"]),
                width=350
            )
        except:
            st.image("https://via.placeholder.com/350x250?text=No+Image")

    with col_text:
        st.markdown(f"### {main_rec['Name'].title()}")
        st.write(f"**Kategori:** {main_rec['C_Type']}")
        st.write(f"**Tipe:** {main_rec['Veg_Non']}")
        st.write(f"**Rating:** {main_rec['Rating']}")
        st.write(f"**Deskripsi:** {main_rec['Describe']}")

    st.markdown("---")


    # =====================================================
    # OTHER RECOMMENDATIONS
    # =====================================================
    st.markdown("## 📌 Rekomendasi Lainnya")

    for i, row in others.iterrows():
        st.markdown("----")
        col_img, col_text = st.columns([1, 2])

        with col_img:
            try:
                st.image(
                    get_food_image(row["Name"]),
                    width=250
                )
            except:
                st.image("https://via.placeholder.com/250x180?text=No+Image")

        with col_text:
            st.markdown(f"### {row['Name'].title()}")
            st.write(f"**Rating:** {row['Rating']}")
            st.write(row["Describe"])
