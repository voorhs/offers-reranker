"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from training.config import Config, DataConfig, ModelConfig, TrainingConfig, load_config


def test_data_config_valid():
    """Test DataConfig with valid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.json"
        dev_path = Path(tmpdir) / "dev.json"
        train_path.write_text("[]")
        dev_path.write_text("[]")

        config = DataConfig(
            train_path=train_path,
            dev_path=dev_path,
            max_seq_length=256,
        )

        assert config.train_path == train_path
        assert config.dev_path == dev_path
        assert config.max_seq_length == 256


def test_data_config_missing_file():
    """Test DataConfig validation fails with non-existent file."""
    with pytest.raises(ValueError, match="Path does not exist"):
        DataConfig(
            train_path=Path("/nonexistent/train.json"),
            dev_path=Path("/nonexistent/dev.json"),
        )


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()

    assert config.model_name == "cointegrated/rubert-tiny2"
    assert config.num_labels == 1
    assert config.max_length is None


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(output_dir=Path(tmpdir))

        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.warmup_ratio == 0.1
        assert config.eval_strategy == "steps"
        assert config.save_strategy == "steps"


def test_config_integration():
    """Test full Config integration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.json"
        dev_path = Path(tmpdir) / "dev.json"
        output_dir = Path(tmpdir) / "output"

        train_path.write_text("[]")
        dev_path.write_text("[]")

        config = Config(
            data=DataConfig(
                train_path=train_path,
                dev_path=dev_path,
            ),
            model=ModelConfig(model_name="test-model"),
            training=TrainingConfig(
                output_dir=output_dir,
                num_epochs=1,
            ),
        )

        assert config.data.train_path == train_path
        assert config.model.model_name == "test-model"
        assert config.training.num_epochs == 1
        assert output_dir.exists()  # Should be created


def test_load_config_from_yaml():
    """Test loading configuration from YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data files
        train_path = tmpdir / "train.json"
        dev_path = tmpdir / "dev.json"
        train_path.write_text("[]")
        dev_path.write_text("[]")

        # Create config YAML
        config_path = tmpdir / "config.yaml"
        config_dict = {
            "data": {
                "train_path": "train.json",  # Relative path
                "dev_path": "dev.json",
                "max_seq_length": 256,
            },
            "model": {
                "model_name": "test-model",
                "num_labels": 1,
            },
            "training": {
                "output_dir": "output",  # Relative path
                "batch_size": 8,
                "learning_rate": 1e-5,
                "num_epochs": 2,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load config
        config = load_config(config_path)

        # Check that paths are resolved relative to config file
        assert config.data.train_path == tmpdir / "train.json"
        assert config.data.dev_path == tmpdir / "dev.json"
        assert config.data.max_seq_length == 256
        assert config.model.model_name == "test-model"
        assert config.training.batch_size == 8
        assert config.training.num_epochs == 2


def test_load_config_missing_file():
    """Test loading config fails with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
