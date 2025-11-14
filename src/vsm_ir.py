from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VSM:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query, k=10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Ambil top-k dokumen
        top_k_idx = scores.argsort()[-k:][::-1]

        return top_k_idx, scores[top_k_idx]

