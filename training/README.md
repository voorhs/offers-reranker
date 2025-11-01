# Training Pipeline for Offers Reranker

This module implements the training pipeline for the offers reranker using CrossEncoder from sentence-transformers.

## Installation

Install dependencies using uv:

```bash
cd training
uv sync
```

For development (includes pytest):

```bash
uv sync --group dev
```

## Configuration

Training is configured via YAML files. See `configs/example.yaml` for a working example.

### Configuration Structure

```yaml
data:
  train_path: ../data/splits/train.json
  dev_path: ../data/splits/dev.json
  test_path: ../data/splits/test.json
  max_seq_length: 512

model:
  model_name: cointegrated/rubert-tiny2
  num_labels: 1

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 1
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1
  eval_strategy: steps
  eval_steps: 500
  save_strategy: steps
  save_steps: 500
  save_total_limit: 2
  output_dir: ../models/rubert-tiny-reranker
  seed: 42
  fp16: false
  bf16: false
  load_best_model_at_end: true
  metric_for_best_model: eval_ndcg@3
  logging_steps: 100
```

## Usage

### Training

Run training with a config file:

```bash
uv run train-reranker configs/example.yaml
```

Or use the Python module directly:

```bash
uv run python -m training.train configs/example.yaml
```

With verbose output:

```bash
uv run train-reranker configs/example.yaml --verbose
```

### Running Tests

Run all tests:

```bash
uv run pytest
```

## Data Format

Training data should be in JSON format:

```json
[
  {
    "user_query": "search query text",
    "offers": [
      ["offer 1 text", 80],
      ["offer 2 text", 60],
      ["offer 3 text", 90]
    ]
  }
]
```

- `user_query`: The user's search query
- `offers`: List of `[offer_text, relevance_score]` pairs where scores are in range 0-100

## Evaluation Metrics

The training pipeline evaluates models using:

- **NDCG@1, NDCG@3, NDCG@5**: Normalized Discounted Cumulative Gain at positions 1, 3, and 5
- **MRR**: Mean Reciprocal Rank (with threshold=50)

Primary metric for best model selection: **NDCG@3**

## Project Structure

```
training/
├── training/
│   ├── __init__.py
│   ├── config.py       # Pydantic configuration models
│   ├── data.py         # Data loading and preprocessing
│   ├── evaluation.py   # NDCG and MRR evaluation
│   ├── model.py        # Model initialization
│   ├── trainer.py      # Main training orchestration
│   └── train.py        # CLI entry point
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   └── test_evaluation.py
├── configs/
│   └── example.yaml
├── pyproject.toml
└── README.md
```

## Implementation Details

### Model

- Uses `CrossEncoder` from sentence-transformers v5
- Default model: `cointegrated/rubert-tiny2` (for quick debugging)
- Loss function: `BCEWithLogitsLoss` (Binary Cross Entropy with logits)

### Data Processing

- Scores are normalized from 0-100 to 0-1 range for training
- Each query-offer pair becomes a separate training sample
- Evaluation data keeps queries grouped for proper ranking metrics

### Training

- Supports mixed precision training (FP16, BF16)
- Warmup with configurable ratio or steps
- Gradient accumulation for effective larger batch sizes
- Periodic evaluation on dev set with ranking metrics
- Checkpoint management with configurable limits
- Best model selection based on NDCG@3

