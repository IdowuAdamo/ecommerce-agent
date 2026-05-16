"""
Evaluation Framework — Task A & Task B metrics.

Task A: ROUGE-1/2/L, BERTScore, Price RMSE, Behavioral Fidelity
Task B: NDCG@10, Hit Rate@K, MRR
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg

logger = logging.getLogger(__name__)


# ── Task B: Ranking Metrics ───────────────────────────────────────────────

def ndcg_at_k(
    relevance_scores: list[list[float]],
    predicted_scores: list[list[float]],
    k: int = 10,
) -> float:
    """
    Compute NDCG@K over multiple queries.

    Args:
        relevance_scores: True relevance scores per query (shape: [n_queries, n_items])
        predicted_scores: Predicted scores per query (shape: [n_queries, n_items])
        k: Cutoff rank

    Returns:
        Mean NDCG@K across all queries.
    """
    scores = []
    for true_rel, pred_rel in zip(relevance_scores, predicted_scores):
        if sum(true_rel) == 0:
            continue
        ndcg = sklearn_ndcg(
            [true_rel],
            [pred_rel],
            k=k,
        )
        scores.append(ndcg)
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(
    relevant_items: list[list[str]],
    recommended_items: list[list[str]],
    k: int = 10,
) -> float:
    """
    Compute Hit Rate@K.

    A "hit" is when at least one relevant item appears in the top-K.

    Args:
        relevant_items: Ground-truth relevant item IDs per query
        recommended_items: Recommended item IDs per query (ordered)

    Returns:
        Hit Rate@K (fraction of queries with at least one hit)
    """
    hits = 0
    for rel, rec in zip(relevant_items, recommended_items):
        if any(item in rel for item in rec[:k]):
            hits += 1
    return hits / len(relevant_items) if relevant_items else 0.0


def mean_reciprocal_rank(
    relevant_items: list[list[str]],
    recommended_items: list[list[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns:
        MRR across all queries.
    """
    rr_sum = 0.0
    for rel, rec in zip(relevant_items, recommended_items):
        for rank, item in enumerate(rec, start=1):
            if item in rel:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(relevant_items) if relevant_items else 0.0


# ── Task A: Generation Metrics ────────────────────────────────────────────

def compute_rouge_scores(
    generated_reviews: list[str],
    reference_reviews: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.

    Args:
        generated_reviews: Model-generated review texts
        reference_reviews: Ground-truth reference reviews

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores (averaged).
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1_scores, r2_scores, rL_scores = [], [], []
        for gen, ref in zip(generated_reviews, reference_reviews):
            scores = scorer.score(ref, gen)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rL_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": float(np.mean(r1_scores)),
            "rouge2": float(np.mean(r2_scores)),
            "rougeL": float(np.mean(rL_scores)),
        }
    except ImportError:
        logger.warning("rouge_score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bert_score(
    generated_reviews: list[str],
    reference_reviews: list[str],
    lang: str = "en",
) -> dict[str, float]:
    """
    Compute BERTScore Precision, Recall, F1.

    Returns:
        Dict with bert_precision, bert_recall, bert_f1 (averaged).
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            generated_reviews, reference_reviews, lang=lang, verbose=False
        )
        return {
            "bert_precision": float(P.mean()),
            "bert_recall": float(R.mean()),
            "bert_f1": float(F1.mean()),
        }
    except ImportError:
        logger.warning("bert_score not installed. Run: pip install bert-score")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def compute_price_rmse(
    actual_prices: list[float],
    predicted_prices: list[float],
) -> dict[str, float]:
    """
    Compute price prediction RMSE and MAE in Naira.

    Returns:
        Dict with rmse, mae, mape.
    """
    actual = np.array(actual_prices)
    predicted = np.array(predicted_prices)
    errors = actual - predicted
    rmse = float(math.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / (actual + 1e-9))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ── Unified Evaluation Runner ─────────────────────────────────────────────

class EvaluationRunner:
    """Runs full Task A and Task B evaluation suites."""

    def evaluate_task_a(
        self,
        generated_reviews: list[str],
        reference_reviews: list[str],
        actual_prices: Optional[list[float]] = None,
        predicted_prices: Optional[list[float]] = None,
    ) -> dict:
        """Full Task A evaluation: ROUGE + BERTScore + RMSE."""
        results: dict = {}
        results.update(compute_rouge_scores(generated_reviews, reference_reviews))
        results.update(compute_bert_score(generated_reviews, reference_reviews))
        if actual_prices and predicted_prices:
            results.update(compute_price_rmse(actual_prices, predicted_prices))
        return results

    def evaluate_task_b(
        self,
        relevance_scores: list[list[float]],
        predicted_scores: list[list[float]],
        relevant_items: list[list[str]],
        recommended_items: list[list[str]],
        k: int = 10,
    ) -> dict:
        """Full Task B evaluation: NDCG@K + HitRate + MRR."""
        return {
            f"ndcg@{k}": ndcg_at_k(relevance_scores, predicted_scores, k),
            f"hit_rate@{k}": hit_rate_at_k(relevant_items, recommended_items, k),
            "mrr": mean_reciprocal_rank(relevant_items, recommended_items),
        }
