"""
Tests for Task A and Task B evaluation metrics.
"""
import pytest
from app.evaluation.evaluator import (
    ndcg_at_k, hit_rate_at_k, mean_reciprocal_rank, compute_price_rmse
)


def test_ndcg_at_k_perfect():
    """Perfect ranking should give NDCG = 1.0."""
    relevance = [[1.0, 0.5, 0.0, 0.0]]
    predicted = [[0.9, 0.6, 0.2, 0.1]]
    score = ndcg_at_k(relevance, predicted, k=4)
    assert score > 0.9


def test_ndcg_at_k_worst():
    """Reversed ranking should give lower NDCG."""
    relevance = [[1.0, 0.5, 0.0, 0.0]]
    predicted = [[0.1, 0.2, 0.6, 0.9]]  # worst order
    score = ndcg_at_k(relevance, predicted, k=4)
    assert score < 0.8


def test_hit_rate_at_k():
    relevant = [["p1", "p2"], ["p3"]]
    recommended = [["p1", "p5", "p6"], ["p7", "p8"]]
    hr = hit_rate_at_k(relevant, recommended, k=3)
    assert hr == 0.5  # Only first query has a hit


def test_hit_rate_all_hits():
    relevant = [["p1"], ["p2"]]
    recommended = [["p1"], ["p2"]]
    hr = hit_rate_at_k(relevant, recommended, k=1)
    assert hr == 1.0


def test_mrr():
    relevant = [["p2", "p3"], ["p1"]]
    recommended = [["p1", "p2", "p3"], ["p3", "p1", "p2"]]
    mrr = mean_reciprocal_rank(relevant, recommended)
    # Query 1: p2 at rank 2 → 0.5. Query 2: p1 at rank 2 → 0.5
    assert abs(mrr - 0.5) < 0.01


def test_price_rmse():
    actual = [100_000, 200_000, 300_000]
    predicted = [105_000, 190_000, 295_000]
    metrics = compute_price_rmse(actual, predicted)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert "mape" in metrics


def test_price_rmse_perfect():
    actual = [100_000, 200_000]
    predicted = [100_000, 200_000]
    metrics = compute_price_rmse(actual, predicted)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
