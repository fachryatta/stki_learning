import re


class BooleanIR:
    def __init__(self, documents):
        self.documents = documents
        self.index = self.build_index(documents)

    # Build inverted index
    def build_index(self, documents):
        index = {}
        for doc_id, text in enumerate(documents):
            tokens = re.findall(r"\b\w+\b", text.lower())
            for t in set(tokens):
                if t not in index:
                    index[t] = set()
                index[t].add(doc_id)
        return index

    # Query processing
    def search(self, query):
        tokens = query.lower().split()
        result = set(range(len(self.documents)))

        current_op = "AND"

        for token in tokens:
            if token in ["and", "or", "not"]:
                current_op = token.upper()
                continue

            postings = self.index.get(token, set())

            if current_op == "AND":
                result = result & postings
            elif current_op == "OR":
                result = result | postings
            elif current_op == "NOT":
                result = result - postings

        return sorted(list(result))
