"""Tests for the TorchServe reranker handler."""

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from handler import RerankerHandler


class TestRerankerHandlerPostprocess:
    """Tests for postprocessing in RerankerHandler."""

    @pytest.fixture
    def handler(self) -> "RerankerHandler":
        """Create a handler instance for testing."""
        from handler import RerankerHandler

        h = RerankerHandler()
        return h

    def test_postprocess_results(self, handler: "RerankerHandler") -> None:
        """Test postprocessing converts results to JSON strings."""
        data = [
            {
                "results": [
                    {"text": "offer1", "score": 0.9, "rank": 1},
                    {"text": "offer2", "score": 0.8, "rank": 2},
                ]
            }
        ]
        result = handler.postprocess(data)
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert "results" in parsed
        assert len(parsed["results"]) == 2

    def test_postprocess_error(self, handler: "RerankerHandler") -> None:
        """Test postprocessing error response."""
        data = [{"error": "Something went wrong"}]
        result = handler.postprocess(data)
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert "error" in parsed

    def test_postprocess_cyrillic(self, handler: "RerankerHandler") -> None:
        """Test postprocessing handles Cyrillic characters."""
        data = [
            {
                "results": [
                    {"text": "Летнее платье", "score": 0.9, "rank": 1},
                ]
            }
        ]
        result = handler.postprocess(data)
        parsed = json.loads(result[0])
        assert parsed["results"][0]["text"] == "Летнее платье"
