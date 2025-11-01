"""CLI entry point for training the offers reranker."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from training.config import load_config
from training.trainer import OffersRerankerTrainer


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

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

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

        logger.info("\n" + "=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {config.training.output_dir}")
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
