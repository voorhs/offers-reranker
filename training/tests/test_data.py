"""Tests for data loading and preprocessing."""

import json
import tempfile
from pathlib import Path

from training.data import (
    create_dataset,
    load_offers_data,
    normalize_score,
    prepare_cross_encoder_samples,
    prepare_evaluation_data,
)


def test_normalize_score():
    """Test score normalization from 0-100 to 0-1."""
    assert normalize_score(0, 0, 100) == 0.0
    assert normalize_score(100, 0, 100) == 1.0
    assert normalize_score(50, 0, 100) == 0.5
    assert normalize_score(25, 0, 100) == 0.25
    assert normalize_score(75, 0, 100) == 0.75


def test_load_offers_data():
    """Test loading offers data from JSON file."""
    data = [
        {
            "user_query": "test query 1",
            "offers": [
                ["offer 1", 80],
                ["offer 2", 60],
            ],
        },
        {
            "user_query": "test query 2",
            "offers": [
                ["offer 3", 90],
            ],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        loaded_data = load_offers_data(temp_path)
        assert len(loaded_data) == 2
        assert loaded_data[0]["user_query"] == "test query 1"
        assert len(loaded_data[0]["offers"]) == 2
        assert loaded_data[1]["user_query"] == "test query 2"
        assert len(loaded_data[1]["offers"]) == 1
    finally:
        temp_path.unlink()


def test_prepare_cross_encoder_samples():
    """Test conversion to CrossEncoder training format."""
    data = [
        {
            "user_query": "test query",
            "offers": [
                ["offer 1", 100],
                ["offer 2", 50],
                ["offer 3", 0],
            ],
        },
    ]

    samples = prepare_cross_encoder_samples(data)

    # Should have 3 samples (one per offer)
    assert len(samples) == 3

    # Check first sample
    assert samples[0]["sentence1"] == "test query"
    assert samples[0]["sentence2"] == "offer 1"
    assert samples[0]["label"] == 1.0  # 100 -> 1.0

    # Check second sample
    assert samples[1]["sentence1"] == "test query"
    assert samples[1]["sentence2"] == "offer 2"
    assert samples[1]["label"] == 0.5  # 50 -> 0.5

    # Check third sample
    assert samples[2]["sentence1"] == "test query"
    assert samples[2]["sentence2"] == "offer 3"
    assert samples[2]["label"] == 0.0  # 0 -> 0.0


def test_prepare_cross_encoder_samples_multiple_queries():
    """Test preparation with multiple queries."""
    data = [
        {
            "user_query": "query 1",
            "offers": [
                ["offer 1", 80],
                ["offer 2", 60],
            ],
        },
        {
            "user_query": "query 2",
            "offers": [
                ["offer 3", 90],
            ],
        },
    ]

    samples = prepare_cross_encoder_samples(data)

    # Should have 3 total samples
    assert len(samples) == 3

    # Check queries are preserved
    assert samples[0]["sentence1"] == "query 1"
    assert samples[1]["sentence1"] == "query 1"
    assert samples[2]["sentence1"] == "query 2"


def test_prepare_evaluation_data():
    """Test preparation of evaluation data with grouped queries."""
    data = [
        {
            "user_query": "test query 1",
            "offers": [
                ["offer 1", 80],
                ["offer 2", 60],
            ],
        },
        {
            "user_query": "test query 2",
            "offers": [
                ["offer 3", 90],
            ],
        },
    ]

    eval_data = prepare_evaluation_data(data)

    # Should have 2 evaluation samples (one per query)
    assert len(eval_data) == 2

    # Check first sample
    assert eval_data[0]["query"] == "test query 1"
    assert eval_data[0]["texts"] == ["offer 1", "offer 2"]
    assert eval_data[0]["scores"] == [80.0, 60.0]

    # Check second sample
    assert eval_data[1]["query"] == "test query 2"
    assert eval_data[1]["texts"] == ["offer 3"]
    assert eval_data[1]["scores"] == [90.0]


def test_create_dataset():
    """Test creating HuggingFace Dataset."""
    data = [
        {
            "user_query": "test query",
            "offers": [
                ["offer 1", 80],
                ["offer 2", 60],
            ],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        dataset = create_dataset(temp_path)

        # Check dataset properties
        assert len(dataset) == 2
        assert "sentence1" in dataset.column_names
        assert "sentence2" in dataset.column_names
        assert "label" in dataset.column_names

        # Check first sample
        assert dataset[0]["sentence1"] == "test query"
        assert dataset[0]["sentence2"] == "offer 1"
        assert dataset[0]["label"] == 0.8

    finally:
        temp_path.unlink()
