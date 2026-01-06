"""CLI entry point for training the offers reranker."""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from loguru import logger

from training.config import load_config
from training.trainer import OffersRerankerTrainer
from dotenv import load_dotenv


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train offers reranker using CrossEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="offers-reranker",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Setup MLflow tracking URI (defaults to local mlruns/)
        load_dotenv()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        mlflow.set_experiment(args.experiment_name)

        # Enable autologging for transformers (works with HF Trainer)
        mlflow.transformers.autolog(log_models=False)

        with mlflow.start_run(run_name=f"train-{config.model.model_name}"):
            # Log parameters (autolog handles training args like learning_rate, etc.)
            mlflow.log_params(
                {
                    # Model params
                    "model_name": config.model.model_name,
                    "num_labels": config.model.num_labels,
                    # Data params
                    "train_path": str(config.data.train_path),
                    "dev_path": str(config.data.dev_path),
                    "max_seq_length": config.data.max_seq_length,
                    # Training params (not covered by autolog)
                    "seed": config.training.seed,
                }
            )

            # Set tags for organization
            mlflow.set_tags(
                {
                    "model_type": "cross-encoder",
                    "task": "reranking",
                }
            )

            if args.verbose:
                logger.info("\nConfiguration:")
                logger.info(f"  Model: {config.model.model_name}")
                logger.info(f"  Training data: {config.data.train_path}")
                logger.info(f"  Dev data: {config.data.dev_path}")
                logger.info(f"  Output dir: {config.training.output_dir}")
                logger.info(f"  Epochs: {config.training.num_epochs}")
                logger.info(f"  Batch size: {config.training.batch_size}")
                logger.info(f"  Learning rate: {config.training.learning_rate}")
                logger.info("")

            # Initialize trainer
            trainer = OffersRerankerTrainer(config)

            # Run training pipeline
            trainer.run()

            # Log artifacts
            mlflow.log_artifact(str(args.config), artifact_path="config")

            # Log dvc.lock if exists
            dvc_lock = Path("../dvc.lock")
            if dvc_lock.exists():
                mlflow.log_artifact(str(dvc_lock), artifact_path="dvc")

            # Log the trained model directory
            model_path = config.training.output_dir
            if model_path.exists():
                mlflow.log_artifacts(str(model_path), artifact_path="model")

            logger.info("\n" + "=" * 80)
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model saved to: {config.training.output_dir}")
            if run := mlflow.active_run():
                logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info("=" * 80)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
