"""Simple batch prediction script for offers reranker."""

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder


def load_model(model_path: str) -> CrossEncoder:
    """Load the cross-encoder model."""
    print(f"Loading model from {model_path}...")
    model = CrossEncoder(model_path)
    print(f"Model loaded on device: {model.device}")
    return model


def predict_sample(
    model: CrossEncoder,
    query: str,
    offers: list[list],
    batch_size: int = 32,
) -> list[dict]:
    """Predict scores for a single sample and return ranked offers.
    
    Args:
        model: CrossEncoder model
        query: User search query
        offers: List of [text, original_score] pairs
        batch_size: Batch size for inference
        
    Returns:
        List of dicts with text, original_score, predicted_score, rank
    """
    # Extract offer texts
    offer_texts = [offer[0] for offer in offers]
    original_scores = [offer[1] for offer in offers]
    
    # Create query-offer pairs
    pairs = [[query, text] for text in offer_texts]
    
    # Get predictions
    predicted_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    
    # Combine results
    results = []
    for i, (text, orig_score, pred_score) in enumerate(
        zip(offer_texts, original_scores, predicted_scores)
    ):
        results.append({
            "text": text,
            "original_score": orig_score,
            "predicted_score": float(pred_score),
            "original_index": i,
        })
    
    # Sort by predicted score (descending)
    results.sort(key=lambda x: x["predicted_score"], reverse=True)
    
    # Add ranks
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prediction for offers reranker")
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
    
    # Load model
    model = load_model(args.model_path)
    
    # Load input data
    print(f"Loading input from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} samples...")
    
    # Process each sample
    output_data = []
    for i, sample in enumerate(data):
        query = sample["user_query"]
        offers = sample["offers"]
        
        # Get predictions
        ranked_offers = predict_sample(model, query, offers, args.batch_size)
        
        # Build output sample
        output_sample = {
            "user_query": query,
            "predictions": ranked_offers,
        }
        output_data.append(output_sample)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} samples")
    
    # Write output
    print(f"Writing output to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()

