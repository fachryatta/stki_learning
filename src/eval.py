# src/eval.py
import numpy as np


# ============================
#  1. Precision, Recall, F1
# ============================

def precision_recall_f1(retrieved, relevant):
    """
    retrieved : list of docID yg diambil oleh sistem
    relevant  : list docID yg dianggap relevan (gold standard)

    return : precision, recall, f1
    """

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positive = len(retrieved_set & relevant_set)

    precision = true_positive / len(retrieved) if len(retrieved) > 0 else 0
    recall = true_positive / len(relevant) if len(relevant) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# ============================
#  2. MAP@k
# ============================

def map_at_k(retrieved, relevant, k=5):
    """
    Mean Average Precision @ k
    retrieved : list ranking dokumen yang dikembalikan
    relevant  : list dokumen relevan (gold)
    """

    retrieved = retrieved[:k]
    score = 0
    hits = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k)


# ============================
#  3. nDCG@k
# ============================

def dcg_at_k(retrieved, relevant, k):
    dcg = 0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / np.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved, relevant, k=5):
    ideal = sorted(relevant, key=lambda x: 1, reverse=True)
    ideal_dcg = dcg_at_k(ideal, relevant, k)

    if ideal_dcg == 0:
        return 0

    actual_dcg = dcg_at_k(retrieved, relevant, k)
    return actual_dcg / ideal_dcg
