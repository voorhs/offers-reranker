# Predict Container

Lightweight batch prediction container for the offers reranker model.

## Architecture

Uses a **multi-stage Docker build** for minimal image size:

1. **Build stage**: Converts PyTorch model to ONNX format
2. **Runtime stage**: Uses only `onnxruntime` + `tokenizers`

## Build

From repo root:

```bash
docker build -f ./predict/Dockerfile -t offers-predict .
```

## Run

```bash
docker run -v ./data/splits:/data offers-predict \
    --input /data/test.json \
    --output /data/predictions.json
```

## Input Format

JSON array of samples, each containing a user query and list of offers:

```json
[
  {
    "user_query": "белые кеды",
    "offers": [
      ["Категории: Обувь, Кеды\nНазвание: Кеды Yarrow\nБренд: HUGO\n...", 95],
      ["Категории: Обувь, Кеды\nНазвание: Кеды\nБренд: Ganni\n...", 93]
    ]
  }
]
```

Each offer is a `[text, original_score]` pair.

## Output Format

JSON array with predictions for each sample:

```json
[
  {
    "user_query": "белые кеды",
    "predictions": [
      {
        "text": "Категории: Обувь, Кеды\nНазвание: Кеды Yarrow\n...",
        "original_score": 95,
        "predicted_score": 0.987,
        "original_index": 0,
        "rank": 1
      }
    ]
  }
]
```

Predictions are sorted by `predicted_score` (descending), with `rank` starting from 1.
