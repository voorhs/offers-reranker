"""Unit tests for reranker service."""

import pytest
from mosec import ClientError

from reranker.settings import RerankerConfig
from reranker.struct import RankedOffer, RerankRequest, RerankResponse


class TestDataStructures:
    """Test data structures and serialization."""
    
    def test_rerank_request_creation(self) -> None:
        """Test creating a RerankRequest."""
        request = RerankRequest(
            query="summer dress",
            offers=["Blue summer dress", "Winter coat"],
            top_k=1,
        )
        assert request.query == "summer dress"
        assert len(request.offers) == 2
        assert request.top_k == 1
    
    def test_rerank_request_without_top_k(self) -> None:
        """Test creating a RerankRequest without top_k."""
        request = RerankRequest(
            query="summer dress",
            offers=["Blue summer dress"],
        )
        assert request.top_k is None
    
    def test_ranked_offer_creation(self) -> None:
        """Test creating a RankedOffer."""
        offer = RankedOffer(
            text="Blue summer dress",
            score=0.95,
            rank=1,
        )
        assert offer.text == "Blue summer dress"
        assert offer.score == 0.95
        assert offer.rank == 1
    
    def test_rerank_response_creation(self) -> None:
        """Test creating a RerankResponse."""
        offers = [
            RankedOffer(text="Offer 1", score=0.9, rank=1),
            RankedOffer(text="Offer 2", score=0.8, rank=2),
        ]
        response = RerankResponse(results=offers)
        assert len(response.results) == 2
        assert response.results[0].score > response.results[1].score


class TestSettings:
    """Test configuration settings."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RerankerConfig()
        assert config.model_path == "../models/rubert-tiny-reranker"
        assert config.batch_size == 32
        assert config.max_offers == 100
        assert config.max_query_length == 512
        assert config.max_offer_length == 1000
        assert config.device == "auto"
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RerankerConfig(
            model_path="custom/path",
            batch_size=64,
            max_offers=50,
        )
        assert config.model_path == "custom/path"
        assert config.batch_size == 64
        assert config.max_offers == 50


class TestValidation:
    """Test input validation logic."""
    
    def test_empty_query(self) -> None:
        """Test that empty query is invalid."""
        from reranker.worker import RerankerWorker
        
        # We can't easily mock the model loading, so we'll test validation separately
        # by creating a mock worker
        request = RerankRequest(query="", offers=["offer1"])
        
        # Validation should fail for empty query
        with pytest.raises(ClientError, match="Query cannot be empty"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_empty_offers(self) -> None:
        """Test that empty offers list is invalid."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(query="test query", offers=[])
        
        with pytest.raises(ClientError, match="Offers list cannot be empty"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_too_many_offers(self) -> None:
        """Test that too many offers is invalid."""
        from reranker.worker import RerankerWorker
        
        config = RerankerConfig(max_offers=10)
        request = RerankRequest(
            query="test",
            offers=[f"offer{i}" for i in range(20)],
        )
        
        with pytest.raises(ClientError, match="exceeds maximum allowed"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = config
            worker._validate_request(request)
    
    def test_invalid_top_k_zero(self) -> None:
        """Test that zero top_k is invalid."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(
            query="test",
            offers=["offer1", "offer2"],
            top_k=0,
        )
        
        with pytest.raises(ClientError, match="top_k must be positive"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_invalid_top_k_exceeds_offers(self) -> None:
        """Test that top_k exceeding offers count is invalid."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(
            query="test",
            offers=["offer1", "offer2"],
            top_k=5,
        )
        
        with pytest.raises(ClientError, match="cannot exceed number of offers"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_query_too_long(self) -> None:
        """Test that too long query is invalid."""
        from reranker.worker import RerankerWorker
        
        config = RerankerConfig(max_query_length=10)
        request = RerankRequest(
            query="a" * 20,
            offers=["offer1"],
        )
        
        with pytest.raises(ClientError, match="Query length.*exceeds maximum"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = config
            worker._validate_request(request)
    
    def test_offer_too_long(self) -> None:
        """Test that too long offer is invalid."""
        from reranker.worker import RerankerWorker
        
        config = RerankerConfig(max_offer_length=10)
        request = RerankRequest(
            query="test",
            offers=["a" * 20],
        )
        
        with pytest.raises(ClientError, match="Offer at index 0 length.*exceeds maximum"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = config
            worker._validate_request(request)
    
    def test_empty_offer(self) -> None:
        """Test that empty offer string is invalid."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(
            query="test",
            offers=["offer1", "", "offer3"],
        )
        
        with pytest.raises(ClientError, match="Offer at index 1 cannot be empty"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_whitespace_only_offer(self) -> None:
        """Test that whitespace-only offer is invalid."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(
            query="test",
            offers=["offer1", "   ", "offer3"],
        )
        
        with pytest.raises(ClientError, match="Offer at index 1 cannot be empty"):
            worker = RerankerWorker.__new__(RerankerWorker)
            worker.config = RerankerConfig()
            worker._validate_request(request)
    
    def test_valid_request(self) -> None:
        """Test that valid request passes validation."""
        from reranker.worker import RerankerWorker
        
        request = RerankRequest(
            query="summer dress",
            offers=["Blue summer dress", "Winter coat", "Spring jacket"],
            top_k=2,
        )
        
        # Should not raise
        worker = RerankerWorker.__new__(RerankerWorker)
        worker.config = RerankerConfig()
        worker._validate_request(request)

