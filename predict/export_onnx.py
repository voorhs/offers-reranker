"""Export CrossEncoder model to ONNX format for lightweight inference."""

import argparse
import logging
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(model_path: str, output_dir: str) -> None:
    """Export a HuggingFace model to ONNX format using optimum.

    Args:
        model_path: Path to the HuggingFace model directory
        output_dir: Directory for the output ONNX model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading and exporting model from {model_path}...")

    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True,
    )

    logger.info(f"Saving ONNX model to {output_path}...")
    model.save_pretrained(output_path)

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for ONNX model"
    )
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_dir)
