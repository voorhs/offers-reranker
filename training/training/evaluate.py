"""Standalone evaluation script for DVC pipeline."""

import argparse
import json
import os
from pathlib import Path

import mlflow
from loguru import logger
from sentence_transformers import CrossEncoder

from training.config import load_config
from training.data import load_offers_data, prepare_evaluation_data
from training.evaluation import CrossEncoderRankingEvaluator
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output metrics JSON")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="offers-reranker",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup MLflow tracking URI (defaults to local mlruns/)
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    mlflow.set_experiment(args.experiment_name)

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

    # Run evaluation and log to MLflow
    with mlflow.start_run(run_name=f"evaluate-{config.model.model_name}"):
        # Log parameters
        mlflow.log_params(
            {
                "model_path": str(model_path),
                "test_path": str(test_path),
                "model_name": config.model.model_name,
            }
        )

        # Set tags
        mlflow.set_tags(
            {
                "model_type": "cross-encoder",
                "task": "evaluation",
            }
        )

        # Run evaluation
        evaluator(model)
        metrics = evaluator.last_metrics

        # Log test metrics to MLflow
        mlflow.log_metrics(metrics)

        # Save metrics to file
        logger.info(f"Saving metrics to {args.output}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)

        # Log metrics.json as artifact
        mlflow.log_artifact(str(args.output), artifact_path="metrics")

        logger.info("Evaluation complete!")
        logger.info(f"Metrics: {metrics}")
        if run := mlflow.active_run():
            logger.info(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
