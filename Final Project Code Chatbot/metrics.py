import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json

# -------------------- Evaluation Metrics --------------------
def precision_at_k(y_true, y_pred, k):
    y_true_k = y_true[:k]
    y_pred_k = y_pred[:k]
    return precision_score(y_true_k, y_pred_k, average='binary')

def recall_at_k(y_true, y_pred, k):
    y_true_k = y_true[:k]
    y_pred_k = y_pred[:k]
    return recall_score(y_true_k, y_pred_k, average='binary')

def mean_reciprocal_rank(y_true, y_pred):
    rr = 0
    for i, label in enumerate(y_true):
        if label == 1:
            rr += 1 / (i + 1)
    return rr / len(y_true)

def ndcg_at_k(y_true, y_pred, k):
    ideal_order = np.sort(y_true)[::-1]
    gain = np.array([rel / np.log2(i + 2) for i, rel in enumerate(y_true[:k])])
    ideal_gain = np.array([rel / np.log2(i + 2) for i, rel in enumerate(ideal_order[:k])])
    return np.sum(gain) / np.sum(ideal_gain)

def mean_average_precision(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def bleu_score(reference, hypothesis):
    return sentence_bleu([reference], hypothesis)

def rouge_score(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def ctr(clicks, impressions):
    return clicks / impressions if impressions > 0 else 0

def conversion_rate(conversions, visits):
    return conversions / visits if visits > 0 else 0

# -------------------- Store Metrics to File --------------------
def save_metrics(metrics, filename="metrics_log.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filename="metrics_log.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}