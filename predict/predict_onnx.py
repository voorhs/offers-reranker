"""Lightweight batch prediction using ONNX Runtime - no PyTorch dependency."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation."""
    return 1 / (1 + np.exp(-x))


class ONNXReranker:
    """Lightweight reranker using ONNX Runtime."""

    def __init__(self, model_dir: str, max_length: int = 128):
        """Initialize the ONNX reranker.

        Args:
            model_dir: Directory containing model.onnx and tokenizer.json
            max_length: Maximum sequence length
        """
        model_path = Path(model_dir)
        onnx_path = model_path / "model.onnx"
        tokenizer_path = model_path / "tokenizer.json"

        logger.info(f"Loading ONNX model from {onnx_path}...")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(length=max_length, pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.max_length = max_length

        logger.info("Model loaded successfully!")

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> np.ndarray:
        """Predict relevance scores for query-document pairs.

        Args:
            pairs: List of [query, document] pairs
            batch_size: Batch size for inference

        Returns:
            Array of relevance scores (after sigmoid)
        """
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            encodings = self.tokenizer.encode_batch(
                [(pair[0], pair[1]) for pair in batch]
            )

            input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
            attention_mask = np.array(
                [enc.attention_mask for enc in encodings], dtype=np.int64
            )
            token_type_ids = np.array(
                [enc.type_ids for enc in encodings], dtype=np.int64
            )

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            logits = outputs[0]
            scores = sigmoid(logits[:, 0])
            all_scores.extend(scores.tolist())

        return np.array(all_scores)


def predict_sample(
    model: ONNXReranker,
    query: str,
    offers: list[list],
    batch_size: int = 32,
) -> list[dict]:
    """Predict scores for a single sample and return ranked offers.

    Args:
        model: ONNXReranker instance
        query: User search query
        offers: List of [text, original_score] pairs
        batch_size: Batch size for inference

    Returns:
        List of dicts with text, original_score, predicted_score, rank
    """
    offer_texts = [offer[0] for offer in offers]
    original_scores = [offer[1] for offer in offers]

    pairs = [[query, text] for text in offer_texts]

    predicted_scores = model.predict(pairs, batch_size=batch_size)

    results = []
    for i, (text, orig_score, pred_score) in enumerate(
        zip(offer_texts, original_scores, predicted_scores)
    ):
        results.append(
            {
                "text": text,
                "original_score": orig_score,
                "predicted_score": float(pred_score),
                "original_index": i,
            }
        )

    # Sort by predicted score (descending)
    results.sort(key=lambda x: x["predicted_score"], reverse=True)

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch prediction for offers reranker (ONNX)"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/model",
        help="Path to model directory (default: /model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    args = parser.parse_args()

    model = ONNXReranker(args.model_path)

    logger.info(f"Loading input from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} samples...")

    output_data = []
    for i, sample in enumerate(data):
        query = sample["user_query"]
        offers = sample["offers"]

        ranked_offers = predict_sample(model, query, offers, args.batch_size)

        output_sample = {
            "user_query": query,
            "predictions": ranked_offers,
        }
        output_data.append(output_sample)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} samples")

    logger.info(f"Writing output to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
