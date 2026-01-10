#!/usr/bin/env python3
"""Script to create a dummy CrossEncoder model for CI testing."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from sentence_transformers import CrossEncoder


def create_dummy_model(output_dir: Path) -> None:
    """Create a dummy CrossEncoder model with minimal configuration.

    Args:
        output_dir: Directory to save the dummy model
    """
    print(f"Creating dummy model in: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a small, lightweight model as the base
    # We'll use a tiny BERT model that's suitable for cross-encoding
    base_model = "prajjwal1/bert-tiny"

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a CrossEncoder with the tiny model
        # This will be untrained but have the correct structure
        print(f"Downloading base model: {base_model}")
        model = CrossEncoder(base_model)

        # Save the model to temporary directory first
        temp_model_dir = temp_path / "model"
        model.save(str(temp_model_dir))

        # Copy all the required files to the output directory
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]

        print("Copying model files...")
        for file_name in required_files:
            src = temp_model_dir / file_name
            dst = output_dir / file_name

            if src.exists():
                shutil.copy2(src, dst)
                print(f"  ✓ {file_name}")
            else:
                print(f"  ⚠ {file_name} (not found, may be optional)")

        # Also copy any remaining files that might be needed
        for file_path in temp_model_dir.iterdir():
            if file_path.is_file() and not (output_dir / file_path.name).exists():
                shutil.copy2(file_path, output_dir / file_path.name)
                print(f"  ✓ {file_path.name} (additional)")

        print(f"Dummy model created successfully at: {output_dir}")

        # Verify the model can be loaded
        print("Verifying model can be loaded...")
        test_model = CrossEncoder(str(output_dir))
        print("  ✓ Model loads successfully")

        # Test a simple prediction to ensure it works
        test_scores = test_model.predict([("test query", "test document")])
        print(f"  ✓ Model prediction works (score: {test_scores[0]:.4f})")


def main() -> None:
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create a dummy CrossEncoder model for CI testing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dummy-model"),
        help="Output directory for the dummy model (default: dummy-model)",
    )

    args = parser.parse_args()

    create_dummy_model(args.output_dir)


if __name__ == "__main__":
    main()
