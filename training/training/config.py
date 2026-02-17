"""Configuration management using Pydantic models."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data configuration."""

    train_path: Path = Field(..., description="Path to training data JSON file")
    dev_path: Path = Field(..., description="Path to development data JSON file")
    test_path: Optional[Path] = Field(None, description="Path to test data JSON file")
    max_seq_length: int = Field(512, description="Maximum sequence length for tokenization", gt=0)

    @field_validator("train_path", "dev_path", "test_path")
    @classmethod
    def validate_path_exists(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate that paths exist."""
        if v is not None and not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    model_name: str = Field(
        "cointegrated/rubert-tiny2",
        description="HuggingFace model name or path to local model",
    )
    num_labels: int = Field(1, description="Number of labels (1 for regression)", ge=1)
    max_length: Optional[int] = Field(None, description="Max length for model inputs")


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Basic training params
    batch_size: int = Field(16, description="Training batch size", gt=0)
    learning_rate: float = Field(2e-5, description="Learning rate", gt=0)
    num_epochs: int = Field(3, description="Number of training epochs", gt=0)

    # Optimization params
    warmup_steps: Optional[int] = Field(None, description="Number of warmup steps", ge=0)
    warmup_ratio: float = Field(0.1, description="Warmup ratio", ge=0, le=1)
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps", gt=0)
    weight_decay: float = Field(0.01, description="Weight decay", ge=0)
    max_grad_norm: float = Field(1.0, description="Max gradient norm", gt=0)

    # Evaluation params
    eval_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Evaluation strategy")
    eval_steps: Optional[int] = Field(500, description="Evaluation steps", gt=0)

    # Saving params
    save_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Save strategy")
    save_steps: Optional[int] = Field(500, description="Save steps", gt=0)
    save_total_limit: Optional[int] = Field(2, description="Maximum number of checkpoints to keep", ge=1)
    load_best_model_at_end: bool = Field(True, description="Load best model at end of training")
    metric_for_best_model: str = Field(
        "eval_evaluator", description="Metric to use for best model selection (eval_evaluator returns NDCG@3)"
    )

    # Output and logging
    output_dir: Path = Field(..., description="Output directory for model and checkpoints")
    logging_steps: int = Field(100, description="Logging steps", gt=0)

    # Reproducibility
    seed: int = Field(42, description="Random seed", ge=0)

    # Performance
    fp16: bool = Field(False, description="Use FP16 mixed precision training")
    bf16: bool = Field(False, description="Use BF16 mixed precision training")
    dataloader_num_workers: int = Field(0, description="Number of dataloader workers", ge=0)

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class Config(BaseModel):
    """Root configuration combining all config sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(config_path: Path | str) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Resolve paths relative to config file location or current working directory
    config_dir = config_path.parent.resolve()
    cwd = Path.cwd()

    def resolve_path(path_str: str, check_existence: bool = True) -> Path:
        """Resolve a path relative to config directory or CWD.

        If check_existence is True, prefers the path that exists.
        Otherwise tries config directory first, then CWD.
        """
        path = Path(path_str)
        if path.is_absolute():
            return path

        # Try both locations
        resolved_from_config = (config_dir / path).resolve()
        resolved_from_cwd = (cwd / path).resolve()

        if check_existence:
            # Prefer the path that exists
            if resolved_from_cwd.exists():
                return resolved_from_cwd
            elif resolved_from_config.exists():
                return resolved_from_config
            # If neither exists, return CWD-relative (more common case)
            # Validation will catch if it doesn't exist
            return resolved_from_cwd
        else:
            # For output dirs that don't exist yet, prefer config-relative
            return resolved_from_config

    if "data" in config_dict:
        for path_key in ["train_path", "dev_path", "test_path"]:
            if path_key in config_dict["data"] and config_dict["data"][path_key]:
                resolved = resolve_path(config_dict["data"][path_key], check_existence=True)
                config_dict["data"][path_key] = str(resolved)

    if "training" in config_dict and "output_dir" in config_dict["training"]:
        resolved = resolve_path(config_dict["training"]["output_dir"], check_existence=False)
        config_dict["training"]["output_dir"] = str(resolved)

    return Config(**config_dict)
