import argparse
from src.preprocess import load_and_preprocess_food
from src.boolean_ir import BooleanIR
from src.vsm_ir import VSM


# ============================
#   Search Engine Orchestrator
# ============================

class SearchEngine:
    def __init__(self, model_type="vsm", data_path="d:/KULIAH/Semester 5/STKI/stki-uts-A11.2023.15170-Ijlal Fachry Attallah/data/1662574418893344.csv"):
        self.food_df = load_and_preprocess_food(data_path)
        self.documents = self.food_df["text"].tolist()

        if model_type == "boolean":
            self.model = BooleanIR(self.documents)

        elif model_type == "vsm":
            self.model = VSM(self.documents)

        else:
            raise ValueError("Model tidak dikenal: gunakan 'boolean' atau 'vsm'.")

        self.model_type = model_type

    def search(self, query, k=5):
        """Mengembalikan hasil pencarian + skor"""
        if self.model_type == "boolean":
            results = self.model.search(query)
            # boolean return: list of doc_idx
            return [(idx, 1.0) for idx in results[:k]]

        elif self.model_type == "vsm":
            top_idx, scores = self.model.search(query, k)
            return list(zip(top_idx, scores))


# =====================================
#   Fungsi lama tetap dipertahankan
# =====================================

def run_search(model_type, query, k):
    engine = SearchEngine(model_type=model_type)
    return engine.search(query, k)


# =====================================
#   CLI Mode (python -m src.search ...)
# =====================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Search Engine STKI UTS")

    parser.add_argument("--model", type=str, choices=["boolean", "vsm"], required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    engine = SearchEngine(model_type=args.model)
    results = engine.search(args.query, args.k)

    print("\n=== HASIL PENCARIAN ===\n")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank}. DocID: {doc_id} | Score: {score:.4f}")
