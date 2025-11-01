"""Tests for evaluation metrics."""

import numpy as np
import pytest

from training.evaluation import dcg_at_k, ndcg_at_k, mrr


def test_dcg_at_k_perfect():
    """Test DCG with perfect ranking."""
    scores = [100, 90, 80, 70, 60]
    
    # DCG@1 = 100 / log2(2) = 100 / 1 = 100
    assert dcg_at_k(scores, 1) == pytest.approx(100.0)
    
    # DCG@3 = 100/log2(2) + 90/log2(3) + 80/log2(4)
    expected = 100 / np.log2(2) + 90 / np.log2(3) + 80 / np.log2(4)
    assert dcg_at_k(scores, 3) == pytest.approx(expected)


def test_dcg_at_k_empty():
    """Test DCG with empty list."""
    assert dcg_at_k([], 3) == 0.0


def test_dcg_at_k_less_than_k():
    """Test DCG when list has fewer items than k."""
    scores = [100, 90]
    
    # Should compute DCG for all available items
    expected = 100 / np.log2(2) + 90 / np.log2(3)
    assert dcg_at_k(scores, 5) == pytest.approx(expected)


def test_ndcg_at_k_perfect():
    """Test NDCG with perfect ranking (already sorted)."""
    scores = [100, 90, 80, 70, 60]
    
    # Perfect ranking should give NDCG = 1.0
    assert ndcg_at_k(scores, 3) == pytest.approx(1.0)
    assert ndcg_at_k(scores, 5) == pytest.approx(1.0)


def test_ndcg_at_k_worst():
    """Test NDCG with worst ranking (reverse sorted)."""
    scores = [60, 70, 80, 90, 100]
    
    # Calculate expected NDCG
    dcg = dcg_at_k(scores, 3)
    ideal_scores = [100, 90, 80]
    idcg = dcg_at_k(ideal_scores, 3)
    expected = dcg / idcg
    
    result = ndcg_at_k(scores, 3)
    assert result == pytest.approx(expected)
    assert result < 1.0  # Should be less than perfect


def test_ndcg_at_k_partial():
    """Test NDCG with partially correct ranking."""
    scores = [90, 100, 70, 80, 60]  # Not perfectly sorted
    
    # Should give NDCG between 0 and 1
    result = ndcg_at_k(scores, 3)
    assert 0.0 < result < 1.0


def test_ndcg_at_k_all_zeros():
    """Test NDCG when all scores are zero."""
    scores = [0, 0, 0]
    
    # When IDCG is 0, should return 0
    assert ndcg_at_k(scores, 3) == 0.0


def test_mrr_first_relevant():
    """Test MRR when first item is relevant."""
    scores = [80, 40, 30]  # First score >= 50
    
    # Reciprocal rank = 1/1 = 1.0
    assert mrr(scores, threshold=50.0) == pytest.approx(1.0)


def test_mrr_second_relevant():
    """Test MRR when second item is relevant."""
    scores = [40, 80, 30]  # Second score >= 50
    
    # Reciprocal rank = 1/2 = 0.5
    assert mrr(scores, threshold=50.0) == pytest.approx(0.5)


def test_mrr_third_relevant():
    """Test MRR when third item is relevant."""
    scores = [30, 40, 80]  # Third score >= 50
    
    # Reciprocal rank = 1/3 ≈ 0.333
    assert mrr(scores, threshold=50.0) == pytest.approx(1.0 / 3.0)


def test_mrr_no_relevant():
    """Test MRR when no items are relevant."""
    scores = [30, 40, 20]  # All scores < 50
    
    # No relevant items, should return 0
    assert mrr(scores, threshold=50.0) == 0.0


def test_mrr_custom_threshold():
    """Test MRR with custom threshold."""
    scores = [70, 60, 50]
    
    # With threshold=60, first item is relevant
    assert mrr(scores, threshold=60.0) == pytest.approx(1.0)
    
    # With threshold=80, no items are relevant
    assert mrr(scores, threshold=80.0) == 0.0


def test_mrr_at_boundary():
    """Test MRR when score exactly equals threshold."""
    scores = [40, 50, 30]
    
    # Second score = 50 (equals threshold), should be considered relevant
    assert mrr(scores, threshold=50.0) == pytest.approx(0.5)

