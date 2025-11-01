# ETL Pipeline for Offers Reranker

This directory contains the ETL pipeline for processing offers reranker data. The pipeline consists of three main steps:

1. **Load**: Fetch raw data from Logfire
2. **Parse**: Transform raw data into structured format
3. **Split**: Divide data into train/dev/test sets

## Setup

Install dependencies using `uv`:

```bash
uv sync
```

Make sure you have a `.env` file with your Logfire credentials:

```
LOGFIRE_READ_TOKEN=your_token_here
```

## Usage

The `pipeline.py` script provides several commands that can be run independently or as part of a full pipeline.

### Full Pipeline

Run all three steps (load, parse, split) in one command:

```bash
uv run python pipeline.py
```

With custom parameters:

```bash
uv run python pipeline.py \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --batch-size 5000
```

Skip specific steps if you already have intermediate files:

```bash
# Skip loading, use existing samples.jsonl
uv run python pipeline.py full-pipeline --skip-load

# Skip both loading and parsing, only perform splitting
uv run python pipeline.py full-pipeline --skip-load --skip-parse
```

### Individual Commands

#### Load Data

Fetch raw data from Logfire and save to JSONL file:

```bash
uv run python pipeline.py load \
  --output-path data/samples.jsonl \
  --batch-size 10000 \
  --timeout 10
```

#### Parse Data

Parse raw JSONL into structured format:

```bash
uv run python pipeline.py parse \
  --input-path data/samples.jsonl \
  --output-path data/parsed_samples.json \
  --show-example
```

#### Split Data

Split parsed data into train/dev/test sets:

```bash
uv run python pipeline.py split \
  --input-path data/parsed_samples.json \
  --output-dir data/splits \
  --train-ratio 0.7 \
  --dev-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

## Command Reference

### `load`

Fetch data from Logfire.

**Parameters:**
- `--output-path`: Path to output JSONL file (default: `data/samples.jsonl`)
- `--batch-size`: Records per batch (default: 10000)
- `--timeout`: Query timeout in seconds (default: 10)

### `parse`

Parse raw JSONL into structured format.

**Parameters:**
- `--input-path`: Path to input JSONL file (default: `data/samples.jsonl`)
- `--output-path`: Path to output JSON file (default: `data/parsed_samples.json`)
- `--show-example`: Show example of parsed data (default: true)

**Input format** (JSONL):
Each line is a JSON object with:
- `request`: XML-like string with `<UserQuery>` and `<Offer>` tags
- `response`: JSON string with `scores` array

**Output format** (JSON):
JSON array where each element is:
```json
{
  "user_query": "search query text",
  "offers": [
    ["offer text 1", score1],
    ["offer text 2", score2],
    ...
  ]
}
```

### `split`

Split data into train/dev/test sets.

**Parameters:**
- `--input-path`: Path to input parsed JSON (default: `data/parsed_samples.json`)
- `--output-dir`: Directory for split files (default: `data/splits`)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--dev-ratio`: Development set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

**Note:** Ratios must sum to 1.0.

**Output:** Creates three files in the output directory:
- `train.json`
- `dev.json`
- `test.json`

### `full-pipeline`

Run the complete pipeline.

**Parameters:**
- `--raw-output`: Path for raw JSONL file (default: `data/samples.jsonl`)
- `--parsed-output`: Path for parsed JSON file (default: `data/parsed_samples.json`)
- `--split-dir`: Directory for splits (default: `data/splits`)
- `--batch-size`: Records per batch for loading (default: 10000)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--dev-ratio`: Development set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed (default: 42)
- `--skip-load`: Skip loading step (default: false)
- `--skip-parse`: Skip parsing step (default: false)

## Examples

### Process existing raw data

If you already have `samples.jsonl`:

```bash
uv run python pipeline.py full-pipeline --skip-load
```

### Custom split ratios

Create 80/10/10 train/dev/test split:

```bash
uv run python pipeline.py split \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1
```

### Help

Get help for any command:

```bash
uv run python pipeline.py --help
uv run python pipeline.py load --help
uv run python pipeline.py parse --help
uv run python pipeline.py split --help
uv run python pipeline.py full-pipeline --help
```

## Data Flow

```
Logfire
   ↓
[load] → samples.jsonl (raw JSONL)
   ↓
[parse] → parsed_samples.json (structured JSON)
   ↓
[split] → splits/
           ├── train.json
           ├── dev.json
           └── test.json
```
