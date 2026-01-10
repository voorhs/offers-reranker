"""Tests for the TorchServe reranker handler."""

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from handler import RerankerHandler


class TestRerankerHandlerPreprocess:
    """Tests for preprocessing in RerankerHandler."""

    @pytest.fixture
    def handler(self) -> "RerankerHandler":
        """Create a handler instance for testing."""
        from handler import RerankerHandler

        h = RerankerHandler()
        return h

    def test_preprocess_json_bytes(self, handler: "RerankerHandler") -> None:
        """Test preprocessing JSON bytes."""
        request = {
            "body": json.dumps(
                {
                    "query": "test",
                    "offers": ["offer1", "offer2"],
                }
            ).encode("utf-8")
        }
        result = handler.preprocess([request])
        assert len(result) == 1
        assert result[0]["query"] == "test"
        assert result[0]["offers"] == ["offer1", "offer2"]

    def test_preprocess_json_string(self, handler: "RerankerHandler") -> None:
        """Test preprocessing JSON string."""
        request = {
            "body": json.dumps(
                {
                    "query": "test",
                    "offers": ["offer1"],
                }
            )
        }
        result = handler.preprocess([request])
        assert len(result) == 1
        assert result[0]["query"] == "test"

    def test_preprocess_dict_body(self, handler: "RerankerHandler") -> None:
        """Test preprocessing dict body."""
        request = {
            "body": {
                "query": "test",
                "offers": ["offer1"],
            }
        }
        result = handler.preprocess([request])
        assert len(result) == 1
        assert result[0]["query"] == "test"

    def test_preprocess_missing_body(self, handler: "RerankerHandler") -> None:
        """Test preprocessing request without body."""
        request: dict[str, object] = {}
        result = handler.preprocess([request])
        assert len(result) == 1
        assert "error" in result[0]

    def test_preprocess_invalid_json(self, handler: "RerankerHandler") -> None:
        """Test preprocessing invalid JSON."""
        request = {"body": b"not valid json"}
        result = handler.preprocess([request])
        assert len(result) == 1
        assert "error" in result[0]
        assert "Invalid JSON" in result[0]["error"]
