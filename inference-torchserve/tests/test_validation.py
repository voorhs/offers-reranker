"""Tests for the TorchServe reranker handler."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from handler import RerankerHandler


class TestRerankerHandlerValidation:
    """Tests for request validation in RerankerHandler."""

    @pytest.fixture
    def handler(self) -> "RerankerHandler":
        """Create a handler instance for testing."""
        # Import here to avoid issues with missing torchserve in test environment
        from handler import RerankerHandler

        h = RerankerHandler()
        return h

    def test_validate_request_valid(self, handler: "RerankerHandler") -> None:
        """Test validation passes for valid request."""
        data = {
            "query": "test query",
            "offers": ["offer1", "offer2"],
            "top_k": 1,
        }
        assert handler._validate_request(data) is None

    def test_validate_request_valid_without_top_k(self, handler: "RerankerHandler") -> None:
        """Test validation passes without top_k."""
        data = {
            "query": "test query",
            "offers": ["offer1", "offer2"],
        }
        assert handler._validate_request(data) is None

    def test_validate_request_empty_query(self, handler: "RerankerHandler") -> None:
        """Test validation fails for empty query."""
        data = {
            "query": "",
            "offers": ["offer1"],
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "Query cannot be empty" in error

    def test_validate_request_query_too_long(self, handler: "RerankerHandler") -> None:
        """Test validation fails for query exceeding max length."""
        data = {
            "query": "x" * (handler.config.max_query_length + 1),
            "offers": ["offer1"],
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "exceeds maximum" in error

    def test_validate_request_empty_offers(self, handler: "RerankerHandler") -> None:
        """Test validation fails for empty offers list."""
        data = {
            "query": "test query",
            "offers": [],
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "Offers list cannot be empty" in error

    def test_validate_request_too_many_offers(self, handler: "RerankerHandler") -> None:
        """Test validation fails when offers exceed max count."""
        data = {
            "query": "test query",
            "offers": ["offer"] * (handler.config.max_offers + 1),
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "exceeds maximum" in error

    def test_validate_request_offer_too_long(self, handler: "RerankerHandler") -> None:
        """Test validation fails for offer exceeding max length."""
        data = {
            "query": "test query",
            "offers": ["x" * (handler.config.max_offer_length + 1)],
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "exceeds maximum" in error

    def test_validate_request_whitespace_offer(self, handler: "RerankerHandler") -> None:
        """Test validation fails for whitespace-only offer."""
        data = {
            "query": "test query",
            "offers": ["   "],
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "cannot be empty or whitespace" in error

    def test_validate_request_invalid_top_k_type(self, handler: "RerankerHandler") -> None:
        """Test validation fails for non-integer top_k."""
        data = {
            "query": "test query",
            "offers": ["offer1"],
            "top_k": "1",
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "must be an integer" in error

    def test_validate_request_negative_top_k(self, handler: "RerankerHandler") -> None:
        """Test validation fails for negative top_k."""
        data = {
            "query": "test query",
            "offers": ["offer1"],
            "top_k": -1,
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "must be positive" in error

    def test_validate_request_top_k_exceeds_offers(self, handler: "RerankerHandler") -> None:
        """Test validation fails when top_k exceeds offer count."""
        data = {
            "query": "test query",
            "offers": ["offer1", "offer2"],
            "top_k": 5,
        }
        error = handler._validate_request(data)
        assert error is not None
        assert "cannot exceed number of offers" in error
