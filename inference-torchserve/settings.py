"""Configuration settings for TorchServe reranker service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerConfig(BaseSettings):
    """Configuration for the reranker service.

    All settings can be overridden via environment variables with RERANKER_ prefix.
    Example: RERANKER_BATCH_SIZE=64
    """

    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        case_sensitive=False,
    )

    batch_size: int = Field(
        default=32,
        description="Batch size for inference",
        gt=0,
    )

    max_offers: int = Field(
        default=100,
        description="Maximum number of offers allowed per request",
        gt=0,
    )

    max_query_length: int = Field(
        default=512,
        description="Maximum length for query in characters",
        gt=0,
    )

    max_offer_length: int = Field(
        default=1000,
        description="Maximum length for each offer in characters",
        gt=0,
    )

    device: str = Field(
        default="auto",
        description="Device to run inference on ('cuda', 'mps', 'cpu', or 'auto')",
    )


reranker_config = RerankerConfig()
