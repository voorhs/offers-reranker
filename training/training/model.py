"""Model initialization and setup."""

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from training.config import ModelConfig


def create_cross_encoder(config: ModelConfig) -> CrossEncoder:
    """Create and initialize CrossEncoder model.

    Args:
        config: Model configuration

    Returns:
        Initialized CrossEncoder model
    """
    model = CrossEncoder(
        model_name_or_path=config.model_name,
        num_labels=config.num_labels,
        max_length=config.max_length,
    )

    return model


def setup_loss_function(model: CrossEncoder) -> BinaryCrossEntropyLoss:
    """Setup loss function for training.

    Uses BinaryCrossEntropyLoss which:
    - Handles regression with labels in [0, 1]
    - Combines sigmoid activation with BCE for numerical stability
    - Suitable for relevance scoring

    Args:
        model: CrossEncoder model to use with the loss

    Returns:
        Loss function instance
    """
    # BinaryCrossEntropyLoss expects labels in [0, 1] range
    # Our normalized scores are already in this range
    loss = BinaryCrossEntropyLoss(model=model)

    return loss
