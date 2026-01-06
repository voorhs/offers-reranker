"""Standalone evaluation script for DVC pipeline."""

import argparse
import json
from pathlib import Path

from loguru import logger
from sentence_transformers import CrossEncoder

from training.config import load_config
from training.data import load_offers_data, prepare_evaluation_data
from training.evaluation import CrossEncoderRankingEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output metrics JSON")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model from output_dir
    model_path = config.training.output_dir
    logger.info(f"Loading model from {model_path}")
    model = CrossEncoder(str(model_path))

    # Load test data
    test_path = config.data.test_path
    if test_path is None:
        raise ValueError("test_path not specified in config")
    
    logger.info(f"Evaluating on test data: {test_path}")
    test_data_raw = load_offers_data(test_path)
    eval_data = prepare_evaluation_data(test_data_raw)

    # Create evaluator
    evaluator = CrossEncoderRankingEvaluator(
        eval_data=eval_data,
        name="test",
        ndcg_at_k=[1, 3, 5],
        mrr_threshold=50.0,
    )

    # Run evaluation
    evaluator(model)
    metrics = evaluator.last_metrics

    # Save metrics
    logger.info(f"Saving metrics to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete!")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
