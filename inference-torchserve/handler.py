"""Custom TorchServe handler for reranking offers using CrossEncoder model."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from ts.torch_handler.base_handler import BaseHandler  # type: ignore[import-untyped]

from settings import reranker_config

logger = logging.getLogger(__name__)


class RerankerHandler(BaseHandler):  # type: ignore[misc]
    """TorchServe handler for reranking offers using cross-encoder model."""

    model: CrossEncoder

    def __init__(self) -> None:
        """Initialize the handler."""
        super().__init__()
        self.initialized = False
        self.config = reranker_config

    def initialize(self, context: Any) -> None:
        """Initialize the model.

        Args:
            context: TorchServe context containing model information
        """
        logger.info("Initializing RerankerHandler")

        # Get model directory from context
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        if model_dir is None:
            raise RuntimeError("model_dir not found in system properties")

        model_path = Path(model_dir)
        logger.info(f"Loading model from: {model_path}")

        # Validate model path
        self._validate_model_path(model_path)

        # Determine device
        device = self._get_device(properties)
        logger.info(f"Using device: {device}")

        # Load the CrossEncoder model
        try:
            self.model = CrossEncoder(
                model_name=str(model_path),
                device=device,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.initialized = True
        logger.info("RerankerHandler initialized successfully")

    def _validate_model_path(self, model_path: Path) -> None:
        """Validate that the model path contains required files.

        Args:
            model_path: Path to the model directory

        Raises:
            FileNotFoundError: If required files are missing
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not model_path.is_dir():
            raise FileNotFoundError(f"Model path is not a directory: {model_path}")

        required_files = ["config.json"]
        missing_files = [f for f in required_files if not (model_path / f).exists()]

        if missing_files:
            raise FileNotFoundError(f"Model directory is missing required files: {missing_files}")

    def _get_device(self, properties: dict[str, Any]) -> str:
        """Determine the device to use for inference.

        Args:
            properties: System properties from TorchServe context

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        # Check if GPU is available via TorchServe
        gpu_id = properties.get("gpu_id")
        if gpu_id is not None and torch.cuda.is_available():
            return f"cuda:{gpu_id}"

        # Fallback to config setting or auto-detect
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def preprocess(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Preprocess incoming requests.

        Args:
            data: List of request dictionaries

        Returns:
            List of preprocessed request data
        """
        logger.debug(f"Preprocessing {len(data)} requests")

        preprocessed = []
        for request in data:
            # Extract body from request
            body = request.get("body")
            if body is None:
                preprocessed.append({"error": "Request body is missing"})
                continue

            # Parse JSON if needed
            if isinstance(body, (bytes, bytearray)):
                try:
                    body = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as e:
                    preprocessed.append({"error": f"Invalid JSON: {e}"})
                    continue
            elif isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError as e:
                    preprocessed.append({"error": f"Invalid JSON: {e}"})
                    continue

            # Validate the request
            validation_error = self._validate_request(body)
            if validation_error:
                preprocessed.append({"error": validation_error})
                continue

            preprocessed.append(
                {
                    "query": body["query"],
                    "offers": body["offers"],
                    "top_k": body.get("top_k"),
                }
            )

        return preprocessed

    def _validate_request(self, data: dict[str, Any]) -> str | None:
        """Validate the rerank request.

        Args:
            data: Request data dictionary

        Returns:
            Error message if validation fails, None otherwise
        """
        # Validate query
        query = data.get("query")
        if not query:
            return "Query cannot be empty"

        if not isinstance(query, str):
            return f"Query must be a string, got {type(query).__name__}"

        if len(query) > self.config.max_query_length:
            return f"Query length ({len(query)}) exceeds maximum ({self.config.max_query_length})"

        # Validate offers
        offers = data.get("offers")
        if not offers:
            return "Offers list cannot be empty"

        if not isinstance(offers, list):
            return f"Offers must be a list, got {type(offers).__name__}"

        if len(offers) > self.config.max_offers:
            return f"Number of offers ({len(offers)}) exceeds maximum ({self.config.max_offers})"

        # Validate each offer
        for i, offer in enumerate(offers):
            if not isinstance(offer, str):
                return f"Offer at index {i} must be a string, got {type(offer).__name__}"

            if not offer or not offer.strip():
                return f"Offer at index {i} cannot be empty or whitespace only"

            if len(offer) > self.config.max_offer_length:
                return f"Offer at index {i} length ({len(offer)}) exceeds maximum ({self.config.max_offer_length})"

        # Validate top_k if provided
        top_k = data.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int):
                return f"top_k must be an integer, got {type(top_k).__name__}"

            if top_k <= 0:
                return f"top_k must be positive, got {top_k}"

            if top_k > len(offers):
                return f"top_k ({top_k}) cannot exceed number of offers ({len(offers)})"

        return None

    def inference(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inference on preprocessed data.

        Args:
            data: List of preprocessed request dictionaries

        Returns:
            List of inference results
        """
        logger.debug(f"Running inference on {len(data)} requests")

        results = []
        for request in data:
            # Pass through errors from preprocessing
            if "error" in request:
                results.append(request)
                continue

            query = request["query"]
            offers = request["offers"]
            top_k = request.get("top_k")

            # Create query-offer pairs for the model
            pairs = [[query, offer] for offer in offers]

            logger.debug(f"Predicting scores for {len(pairs)} query-offer pairs")

            # Get predictions from model
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

            # Create ranked offers with 1-indexed ranks
            ranked_offers = [
                {
                    "text": offer,
                    "score": score,
                    "rank": rank + 1,  # 1-indexed ranking
                }
                for rank, (offer, score, _) in enumerate(offer_scores)
            ]

            results.append({"results": ranked_offers})

        return results

    def postprocess(self, data: list[dict[str, Any]]) -> list[str]:
        """Postprocess inference results to JSON strings.

        Args:
            data: List of inference result dictionaries

        Returns:
            List of JSON string responses
        """
        logger.debug(f"Postprocessing {len(data)} results")

        responses = []
        for result in data:
            responses.append(json.dumps(result, ensure_ascii=False))

        return responses
