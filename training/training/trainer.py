"""Trainer class for orchestrating the training pipeline."""

import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from datasets import Dataset  # type: ignore[import-untyped]
from training.config import Config
from training.data import create_dataset, load_offers_data, prepare_evaluation_data
from training.evaluation import CrossEncoderRankingEvaluator
from training.model import create_cross_encoder, setup_loss_function


class OffersRerankerTrainer:
    """Orchestrates training of the offers reranker model."""

    def __init__(self, config: Config):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model: CrossEncoder | None = None
        self.trainer: CrossEncoderTrainer | None = None

        # Set random seeds for reproducibility
        self._set_seed(config.training.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_data(self) -> tuple[Dataset, Dataset, CrossEncoderRankingEvaluator]:
        """Load and prepare training and evaluation datasets.

        Returns:
            Tuple of (train_dataset, dev_dataset, evaluator)
        """
        logger.info("Loading datasets...")

        # Load training data
        train_dataset = create_dataset(self.config.data.train_path)
        logger.info(f"Training samples: {len(train_dataset)}")

        # Load development data
        dev_dataset = create_dataset(self.config.data.dev_path)
        logger.info(f"Development samples (flattened): {len(dev_dataset)}")

        # Prepare evaluation data (keeping queries grouped)
        dev_data_raw = load_offers_data(self.config.data.dev_path)
        eval_data = prepare_evaluation_data(dev_data_raw)
        logger.info(f"Development queries (for ranking): {len(eval_data)}")

        # Create evaluator
        evaluator = CrossEncoderRankingEvaluator(
            eval_data=eval_data,
            name="eval",
            ndcg_at_k=[1, 3, 5],
            mrr_threshold=50.0,
        )

        return train_dataset, dev_dataset, evaluator

    def _setup_training(
        self, train_dataset: Dataset, dev_dataset: Dataset, evaluator: CrossEncoderRankingEvaluator
    ) -> None:
        """Setup model, loss, and trainer.

        Args:
            train_dataset: Training dataset
            dev_dataset: Development dataset
            evaluator: Ranking evaluator
        """
        logger.info("\nInitializing model...")
        self.model = create_cross_encoder(self.config.model)

        logger.info("Setting up loss function...")
        loss_fn = setup_loss_function(self.model)

        logger.info("Configuring training arguments...")

        # Build training arguments dict to handle optional parameters
        training_args_dict = {
            # Output
            "output_dir": str(self.config.training.output_dir),
            # Training params
            "num_train_epochs": self.config.training.num_epochs,
            "per_device_train_batch_size": self.config.training.batch_size,
            "per_device_eval_batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "max_grad_norm": self.config.training.max_grad_norm,
            # Optimization
            "warmup_ratio": self.config.training.warmup_ratio,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
            # Evaluation
            "eval_strategy": self.config.training.eval_strategy,
            # Saving
            "save_strategy": self.config.training.save_strategy,
            "save_total_limit": self.config.training.save_total_limit,
            "load_best_model_at_end": self.config.training.load_best_model_at_end,
            "metric_for_best_model": self.config.training.metric_for_best_model,
            # Logging
            "logging_steps": self.config.training.logging_steps,
            # Performance
            "fp16": self.config.training.fp16,
            "bf16": self.config.training.bf16,
            "dataloader_num_workers": self.config.training.dataloader_num_workers,
            # Reproducibility
            "seed": self.config.training.seed,
        }

        # Add optional warmup_steps if specified
        if self.config.training.warmup_steps is not None:
            training_args_dict["warmup_steps"] = self.config.training.warmup_steps

        # Add eval_steps if using steps evaluation
        if self.config.training.eval_strategy == "steps" and self.config.training.eval_steps is not None:
            training_args_dict["eval_steps"] = self.config.training.eval_steps

        # Add save_steps if using steps save strategy
        if self.config.training.save_strategy == "steps" and self.config.training.save_steps is not None:
            training_args_dict["save_steps"] = self.config.training.save_steps

        training_args = CrossEncoderTrainingArguments(**training_args_dict)  # type: ignore[arg-type]

        logger.info("Creating trainer...")
        self.trainer = CrossEncoderTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            loss=loss_fn,
            evaluator=evaluator,
        )

    def _train(self) -> None:
        """Run the training loop."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_training() first.")

        logger.info("\n" + "=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80 + "\n")

        self.trainer.train()

        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80 + "\n")

    def _save_model(self, output_path: Path | str | None = None) -> None:
        """Save the trained model.

        Args:
            output_path: Path to save model (defaults to config output_dir)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        save_path = output_path or self.config.training.output_dir
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving model to {save_path}...")
        self.model.save(str(save_path))
        logger.info("Model saved successfully!")

    def run(self) -> None:
        """Run the complete training pipeline."""
        # Prepare data
        train_dataset, dev_dataset, evaluator = self._prepare_data()

        # Setup training
        self._setup_training(train_dataset, dev_dataset, evaluator)

        # Train
        self._train()

        # Save final model
        self._save_model()
