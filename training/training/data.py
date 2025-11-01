"""Data loading and preprocessing for offers reranker."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_offers_data(data_path: Path) -> list[dict[str, Any]]:
    """Load offers data from JSON file.

    Args:
        data_path: Path to JSON file with format:
            [
                {
                    "user_query": str,
                    "offers": [[offer_text, score], ...]
                },
                ...
            ]

    Returns:
        List of dictionaries containing queries and offers
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def normalize_score(score: float, min_score: float = 0.0, max_score: float = 100.0) -> float:
    """Normalize score from [min_score, max_score] to [0, 1].

    Args:
        score: Original score
        min_score: Minimum possible score
        max_score: Maximum possible score

    Returns:
        Normalized score in [0, 1]
    """
    return (score - min_score) / (max_score - min_score)


def prepare_cross_encoder_samples(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert offers data to CrossEncoder training format.

    Each query-offer pair becomes a separate training sample with normalized score.

    Args:
        data: List of dicts with "user_query" and "offers" keys

    Returns:
        List of samples with format:
            [
                {
                    "sentence1": query,
                    "sentence2": offer_text,
                    "label": normalized_score (0-1)
                },
                ...
            ]
    """
    samples = []

    for item in data:
        user_query = item["user_query"]
        offers = item["offers"]

        for offer_text, score in offers:
            # Normalize score from 0-100 to 0-1
            normalized_score = normalize_score(score, min_score=0.0, max_score=100.0)

            samples.append(
                {
                    "sentence1": user_query,
                    "sentence2": offer_text,
                    "label": normalized_score,
                }
            )

    return samples


def create_dataset(data_path: Path) -> Dataset:
    """Create HuggingFace Dataset from JSON file.

    Args:
        data_path: Path to JSON file

    Returns:
        HuggingFace Dataset ready for CrossEncoder training
    """
    data = load_offers_data(data_path)
    samples = prepare_cross_encoder_samples(data)
    return Dataset.from_list(samples)


def prepare_evaluation_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare data for evaluation with ranking metrics.

    Keeps queries and offers grouped for proper ranking evaluation.

    Args:
        data: List of dicts with "user_query" and "offers" keys

    Returns:
        List of evaluation samples with format:
            [
                {
                    "query": query,
                    "texts": [offer_text1, offer_text2, ...],
                    "scores": [score1, score2, ...]  # Original 0-100 scores
                },
                ...
            ]
    """
    eval_samples = []

    for item in data:
        user_query = item["user_query"]
        offers = item["offers"]

        texts = [offer_text for offer_text, _ in offers]
        scores = [float(score) for _, score in offers]

        eval_samples.append(
            {
                "query": user_query,
                "texts": texts,
                "scores": scores,
            }
        )

    return eval_samples
