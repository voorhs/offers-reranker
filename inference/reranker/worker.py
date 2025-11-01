"""Mosec worker for reranking offers."""

import logging

from mosec import ClientError, Worker
from mosec.mixin import TypedMsgPackMixin
from sentence_transformers import CrossEncoder

from reranker.load import load_cross_encoder
from reranker.settings import reranker_config
from reranker.struct import RankedOffer, RerankRequest, RerankResponse

logger = logging.getLogger(__name__)


class RerankerWorker(TypedMsgPackMixin, Worker):
    """Worker for reranking offers using cross-encoder model."""
    
    def __init__(self) -> None:
        """Initialize the worker and load the model."""
        super().__init__()
        logger.info("Initializing RerankerWorker")
        self.model: CrossEncoder = load_cross_encoder()
        self.config = reranker_config
        logger.info("RerankerWorker initialized successfully")
    
    def forward(self, data: RerankRequest) -> RerankResponse:
        """Process reranking request.
        
        Args:
            data: RerankRequest containing query and offers
            
        Returns:
            RerankResponse with ranked offers
            
        Raises:
            ClientError: If input validation fails
        """
        logger.info(f"Received rerank request with query: '{data.query[:50]}...' and {len(data.offers)} offers")
        
        # Validate input
        self._validate_request(data)
        
        # Perform reranking
        ranked_offers = self._rerank_offers(data.query, data.offers, data.top_k)
        
        response = RerankResponse(results=ranked_offers)
        logger.info(f"Returning {len(ranked_offers)} ranked offers")
        
        return response
    
    def _validate_request(self, data: RerankRequest) -> None:
        """Validate the rerank request.
        
        Args:
            data: RerankRequest to validate
            
        Raises:
            ClientError: If validation fails
        """
        # Validate query
        if not data.query:
            raise ClientError("Query cannot be empty")
        
        if not isinstance(data.query, str):
            raise ClientError(f"Query must be a string, got {type(data.query).__name__}")
        
        if len(data.query) > self.config.max_query_length:
            raise ClientError(
                f"Query length ({len(data.query)}) exceeds maximum allowed length ({self.config.max_query_length})"
            )
        
        # Validate offers
        if not data.offers:
            raise ClientError("Offers list cannot be empty")
        
        if not isinstance(data.offers, list):
            raise ClientError(f"Offers must be a list, got {type(data.offers).__name__}")
        
        if len(data.offers) > self.config.max_offers:
            raise ClientError(
                f"Number of offers ({len(data.offers)}) exceeds maximum allowed ({self.config.max_offers})"
            )
        
        # Validate each offer
        for i, offer in enumerate(data.offers):
            if not isinstance(offer, str):
                raise ClientError(f"Offer at index {i} must be a string, got {type(offer).__name__}")
            
            if not offer or not offer.strip():
                raise ClientError(f"Offer at index {i} cannot be empty or whitespace only")
            
            if len(offer) > self.config.max_offer_length:
                raise ClientError(
                    f"Offer at index {i} length ({len(offer)}) exceeds maximum allowed ({self.config.max_offer_length})"
                )
        
        # Validate top_k if provided
        if data.top_k is not None:
            if not isinstance(data.top_k, int):
                raise ClientError(f"top_k must be an integer, got {type(data.top_k).__name__}")
            
            if data.top_k <= 0:
                raise ClientError(f"top_k must be positive, got {data.top_k}")
            
            if data.top_k > len(data.offers):
                raise ClientError(
                    f"top_k ({data.top_k}) cannot exceed number of offers ({len(data.offers)})"
                )
        
        logger.debug("Request validation passed")
    
    def _rerank_offers(self, query: str, offers: list[str], top_k: int | None) -> list[RankedOffer]:
        """Rerank offers based on query using the cross-encoder model.
        
        Args:
            query: Search query
            offers: List of offer texts
            top_k: Optional limit on number of results
            
        Returns:
            List of RankedOffer objects sorted by relevance (descending)
        """
        # Create query-offer pairs for the model
        pairs = [[query, offer] for offer in offers]
        
        logger.debug(f"Predicting scores for {len(pairs)} query-offer pairs")
        
        # Get predictions from model
        # The model returns scores in the range used during training (0-1 after sigmoid)
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        
        # Create list of (offer, score, original_index) tuples
        offer_scores = [(offers[i], float(scores[i]), i) for i in range(len(offers))]
        
        # Sort by score in descending order (highest relevance first)
        offer_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k filtering if requested
        if top_k is not None:
            offer_scores = offer_scores[:top_k]
        
        # Create RankedOffer objects with 1-indexed ranks
        ranked_offers = [
            RankedOffer(
                text=offer,
                score=score,
                rank=rank + 1,  # 1-indexed ranking
            )
            for rank, (offer, score, _) in enumerate(offer_scores)
        ]
        
        logger.debug(f"Reranking complete. Top score: {ranked_offers[0].score:.4f}")
        
        return ranked_offers

