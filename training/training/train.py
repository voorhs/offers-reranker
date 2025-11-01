"""CLI entry point for training the offers reranker."""

import argparse
import sys
from pathlib import Path

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
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
        
        if args.verbose:
            print("\nConfiguration:")
            print(f"  Model: {config.model.model_name}")
            print(f"  Training data: {config.data.train_path}")
            print(f"  Dev data: {config.data.dev_path}")
            print(f"  Output dir: {config.training.output_dir}")
            print(f"  Epochs: {config.training.num_epochs}")
            print(f"  Batch size: {config.training.batch_size}")
            print(f"  Learning rate: {config.training.learning_rate}")
            print()
        
        # Initialize trainer
        trainer = OffersRerankerTrainer(config)
        
        # Run training pipeline
        trainer.run()
        
        print("\n" + "=" * 80)
        print("Training pipeline completed successfully!")
        print(f"Model saved to: {config.training.output_dir}")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

