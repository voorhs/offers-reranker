#!/bin/bash
# Script to package the reranker model into a TorchServe .mar archive

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$INFERENCE_DIR")"

# Default paths
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/rubert-tiny-reranker}"
OUTPUT_DIR="${OUTPUT_DIR:-$INFERENCE_DIR/model-store}"
MODEL_NAME="${MODEL_NAME:-reranker}"
VERSION="${VERSION:-1.0}"

echo "=== TorchServe Model Packaging ==="
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "Version: $VERSION"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in model path"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Creating temporary packaging directory: $TEMP_DIR"

echo "Copying model files..."
cp -r "$MODEL_PATH"/* "$TEMP_DIR/"

echo "Copying handler and settings..."
cp "$INFERENCE_DIR/handler.py" "$TEMP_DIR/"
cp "$INFERENCE_DIR/settings.py" "$TEMP_DIR/"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    exit 1
fi

# Export requirements.txt from pyproject.toml using uv
echo "Exporting requirements.txt from pyproject.toml..."
uv export --no-hashes --no-dev --no-editable --project "$INFERENCE_DIR" > "$TEMP_DIR/requirements.txt"

if ! command -v uv run torch-model-archiver &> /dev/null; then
    echo "Error: torch-model-archiver is not installed"
    exit 1
fi

MAR_FILE="$OUTPUT_DIR/${MODEL_NAME}.mar"
if [ -f "$MAR_FILE" ]; then
    echo "Removing existing archive: $MAR_FILE"
    rm "$MAR_FILE"
fi

echo "Packaging model with torch-model-archiver..."
uv run torch-model-archiver \
    --model-name "$MODEL_NAME" \
    --version "$VERSION" \
    --handler "$TEMP_DIR/handler.py" \
    --extra-files "$TEMP_DIR/config.json,$TEMP_DIR/model.safetensors,$TEMP_DIR/tokenizer.json,$TEMP_DIR/tokenizer_config.json,$TEMP_DIR/vocab.txt,$TEMP_DIR/special_tokens_map.json,$TEMP_DIR/settings.py" \
    --requirements-file "$TEMP_DIR/requirements.txt" \
    --export-path "$OUTPUT_DIR" \
    --force

echo ""
echo "=== Packaging Complete ==="
echo "Model archive created: $MAR_FILE"
echo ""
echo "To verify the archive, run:"
echo "  unzip -l $MAR_FILE"
echo ""
echo "To build and run with Docker Compose:"
echo "  cd $INFERENCE_DIR && docker compose up --build"

