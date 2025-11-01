"""Data structures for reranker service using msgspec."""

from msgspec import Struct


class RerankRequest(Struct):
    """Request for reranking offers.

    Attributes:
        query: User search query
        offers: List of offer texts to rerank
        top_k: Optional limit for number of results to return
    """

    query: str
    offers: list[str]
    top_k: int | None = None


class RankedOffer(Struct):
    """Single ranked offer result.

    Attributes:
        text: Original offer text
        score: Relevance score from model (0-1 range, higher is more relevant)
        rank: Position in the reranked list (1-indexed)
    """

    text: str
    score: float
    rank: int


class RerankResponse(Struct):
    """Response containing reranked offers.

    Attributes:
        results: List of ranked offers sorted by relevance score (descending)
    """

    results: list[RankedOffer]
