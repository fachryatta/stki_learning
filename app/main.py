import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================
# 1. Load Dataset
# ======================================
@st.cache_data
def load_data():
    food_df = pd.read_csv("data/1662574418893344.csv")
    ratings_df = pd.read_csv("data/ratings.csv")

    if "Food_ID" in food_df.columns and "Food_ID" in ratings_df.columns:
        merged_df = pd.merge(food_df, ratings_df, on="Food_ID", how="inner")
        return merged_df

    st.error("Kolom 'Food_ID' tidak ditemukan pada dataset.")
    st.stop()


df = load_data()


# ======================================
# 2. Preprocessing Text
# ======================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_desc"] = df["Describe"].apply(clean_text)


# ======================================
# 3. TF-IDF Vectorization
# ======================================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_desc"])


# ======================================
# 4. STREAMLIT UI
# ======================================
st.markdown("""
<h1 style='text-align: center; margin-bottom: 40px;'>
🍽️ APLIKASI REKOMENDASI MAKANAN LUAR NEGERI
</h1>
""", unsafe_allow_html=True)

st.write("Pilih filter berikut untuk mendapatkan rekomendasi makanan:")

col1, col2, col3 = st.columns(3)

with col1:
    jenis = st.selectbox(
        "Jenis Makanan:",
        [""] + sorted(df["C_Type"].dropna().unique().tolist())
    )

with col2:
    veg_type = st.selectbox(
        "Vegetarian / Non-Veg:",
        [""] + sorted(df["Veg_Non"].dropna().unique().tolist())
    )

with col3:
    rating_choice = st.slider("Rating minimal:", 1, 10, 5)

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

carikan_clicked = st.button("Carikan Rekomendasi 🍽️")

st.markdown("<hr>", unsafe_allow_html=True)


# ======================================
# 5. Rekomendasi
# ======================================
if carikan_clicked:

    if jenis == "" or veg_type == "":
        st.warning("⚠️ Lengkapi filter terlebih dahulu.")
    else:

        filtered_df = df[
            (df["C_Type"] == jenis) &
            (df["Veg_Non"] == veg_type) &
            (df["Rating"] >= rating_choice)
        ]

        if filtered_df.empty:
            st.warning("Tidak ada makanan yang cocok.")
        else:

            # Hilangkan duplikasi Food_ID
            unique_filtered = (
                filtered_df
                .sort_values("Rating", ascending=False)
                .drop_duplicates("Food_ID")
            )

            idx_unique = unique_filtered.index.tolist()
            tfidf_unique = tfidf_matrix[idx_unique]

            # Hitung mean vector
            avg_vec = tfidf_unique.mean(axis=0).A1

            # Hitung cosine similarity
            similarity_scores = cosine_similarity([avg_vec], tfidf_unique).flatten()

            top_k = min(5, len(similarity_scores))
            top_indices = similarity_scores.argsort()[-top_k:][::-1]

            # ======================
            # Rekomendasi Utama
            # ======================
            best_food = unique_filtered.iloc[top_indices[0]]

            st.markdown(f"""
            <h2>🔥 Rekomendasi Utama</h2>
            <h3>{best_food['Name'].title()}</h3>
            <p><b>Kategori:</b> {best_food['C_Type']} |
            <b>Tipe:</b> {best_food['Veg_Non']}</p>
            <p><b>Rating:</b> {best_food['Rating']}</p>
            <p>{best_food['Describe']}</p>
            """, unsafe_allow_html=True)

            # ======================
            # Rekomendasi Lain
            # ======================
            st.markdown("<hr><h3>📌 Rekomendasi Lainnya</h3>", unsafe_allow_html=True)

            for i, idx in enumerate(top_indices[1:], start=2):
                food = unique_filtered.iloc[idx]
                st.markdown(f"""
                <div style='padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 15px;'>
                    <h4>#{i} — {food['Name'].title()}</h4>
                    <b>Rating:</b> {food['Rating']}<br>
                    <b>Deskripsi:</b> {food['Describe'][:200]}...
                </div>
                """, unsafe_allow_html=True)
