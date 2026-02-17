"""Model loading utilities."""

import logging
from pathlib import Path

from sentence_transformers import CrossEncoder

from reranker.settings import reranker_config

logger = logging.getLogger(__name__)


def validate_model_path(model_path: str | Path) -> Path:
    """Validate that the model path exists and contains required files.

    Args:
        model_path: Path to the model directory

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If model path or required files don't exist
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {path}")

    # Check for required model files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (path / f).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing_files}. "
            f"Please ensure the model is properly saved at {path}"
        )

    logger.info(f"Model path validated: {path}")
    return path


def load_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder model from configured path.

    Returns:
        Loaded CrossEncoder model

    Raises:
        FileNotFoundError: If model path is invalid
        Exception: If model loading fails
    """
    model_path = reranker_config.model_path

    logger.info(f"Loading cross-encoder model from: {model_path}")

    # Validate model path
    validated_path = validate_model_path(model_path)

    try:
        # Load the model
        model = CrossEncoder(
            model_name=str(validated_path),
            device=reranker_config.device,
        )

        logger.info(f"Successfully loaded model on device: {model.device}")
        return model  # type: ignore[no-any-return]

    except Exception as e:
        logger.error(f"Failed to load model from {validated_path}: {e}")
        raise
