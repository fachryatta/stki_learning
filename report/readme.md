APLIKASI REKOMENDASI MAKANAN LUAR NEGERI

Sistem Information Retrieval (IR) menggunakan Boolean Model dan Vector Space Model (VSM) untuk melakukan pencarian makanan berdasarkan deskripsi atau query pengguna.
Project ini terdiri dari preprocessing, indexing, dan search engine sederhana yang dapat dijalankan melalui terminal dan Streamlit.

📁 Struktur Folder
project/
│── data/
│   └── 1662574418893344.csv        # dataset makanan
│
│── src/
│   ├── preprocess.py               # cleaning + TF-IDF text builder
│   ├── boolean_ir.py               # boolean retrieval
│   ├── vsm_ir.py                   # vector space model
│   ├── search.py                   # class SearchEngine
│   └── eval.py                     # evaluasi IR
│
│── notebooks/
│   └── analisis_dataset.ipynb      # eksplorasi + preprocessing
│
│── app/
│   └── app.py                      # versi streamlit (opsional)
│
└── readme.md


📝 Deskripsi Singkat

Aplikasi ini memanfaatkan dua teknik IR:

1. Boolean Retrieval

Menggunakan operator AND, OR, NOT

Menghasilkan dokumen yang cocok secara eksak (exact match)

Cocok digunakan untuk filtering tegas

2. Vector Space Model (VSM)

Menggunakan TF, IDF, TF-IDF, dan Cosine Similarity

Menghasilkan ranking dokumen berdasarkan relevansi

Lebih fleksibel terhadap query natural

⚙️ Cara Menjalankan Program
1. Aktivasi Virtual Environment (opsional tapi disarankan)

python -m venv venv
source venv/Scripts/activate  # Windows

2. Install Dependencies

pip install -r requirements.txt

🚀 Menjalankan Search Engine via Terminal

1. Boolean Search
python src/search.py --model boolean --query "sweet AND chicken" --k 5

2. VSM Search
python src/search.py --model vsm --query "sweet chicken rice" --k 5

🎨 Menjalankan Aplikasi Streamlit (opsional)

Jika ingin menggunakan UI rekomendasi makanan:

streamlit run app/app.py

📌 Asumsi Project

1. Dataset memiliki kolom:
Food_ID, Name, C_Type, Veg_Non, Rating, Describe

2. Preprocessing dilakukan dengan:

- lowercase

- hapus angka dan simbol

- normalisasi spasi

3. Query untuk boolean harus menggunakan operator kapital:
AND, OR, NOT

4. VSM menghitung similarity berdasarkan TF-IDF standar sklearn

📈 Evaluasi

Pada file eval.py tersedia:

- Precision@K
- Recall
- Cosine similarity ranking
- Contoh skenario evaluasi dengan query manual

👤 Author

Ijlal Fachry Attallah
A11.2023.15170
Universitas Dian Nuswantoro
