# Offers Reranker TorchServe Inference Service

TorchServe-based inference service for the offers reranker cross-encoder model.

## Description

This service accepts a user search query and a list of offers, returning offers sorted by relevance using a trained BERT-like cross-encoder model.

## Quick Start

### 1. Install Dependencies

```bash
cd inference-torchserve
uv sync
```

### 2. Package the Model

First, package the model into a TorchServe model archive (.mar):

```bash
chmod +x scripts/package_model.sh
./scripts/package_model.sh
```

This creates `model-store/reranker.mar` containing the model and handler.

### 3. Configure Environment

Copy the example environment file and adjust as needed:

```bash
cp env.example .env
```

### 4. Build and Run with Docker Compose

```bash
docker compose up --build
```

Or build and run separately:

```bash
docker build -t offers-reranker-torchserve .
docker run -p 8080:8080 -p 8081:8081 -p 8082:8082 --env-file .env offers-reranker-torchserve
```

## Configuration

Environment variables for configuring the service:

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_BATCH_SIZE` | `32` | Batch size for inference |
| `RERANKER_MAX_OFFERS` | `100` | Maximum offers per request |
| `RERANKER_MAX_QUERY_LENGTH` | `512` | Maximum query length (chars) |
| `RERANKER_MAX_OFFER_LENGTH` | `1000` | Maximum offer length (chars) |
| `RERANKER_DEVICE` | `auto` | Device for inference (auto/cuda/mps/cpu) |

Configuration is managed via `.env` file (loaded by docker-compose) or environment variables.

## API

### Endpoints

| Port | Endpoint | Description |
|------|----------|-------------|
| 8080 | `/predictions/reranker` | Inference API |
| 8081 | `/models` | Management API |
| 8082 | `/metrics` | Prometheus metrics |

### POST /predictions/reranker

Rerank offers by relevance to a query.

#### Request

```json
{
  "query": "летнее платье",
  "offers": [
    "Синее летнее платье из хлопка",
    "Зимнее пальто с мехом",
    "Легкое платье с цветочным принтом"
  ],
  "top_k": 2
}
```

**Fields:**
- `query` (string, required): User search query
- `offers` (list[string], required): List of offer descriptions to rerank
- `top_k` (integer, optional): Return only top-K most relevant offers

#### Response

```json
{
  "results": [
    {
      "text": "Легкое платье с цветочным принтом",
      "score": 0.9234,
      "rank": 1
    },
    {
      "text": "Синее летнее платье из хлопка",
      "score": 0.8756,
      "rank": 2
    }
  ]
}
```

**Fields:**
- `results` (list): Offers sorted by relevance (descending)
  - `text` (string): Original offer text
  - `score` (float): Relevance score from model (0-1, higher = more relevant)
  - `rank` (integer): Position in reranked list (1-indexed)

## Usage Examples

### cURL

```bash
curl -X POST http://localhost:8080/predictions/reranker \
  -H "Content-Type: application/json" \
  -d '{
    "query": "летнее платье",
    "offers": [
      "Синее летнее платье",
      "Зимнее пальто",
      "Весенняя куртка"
    ],
    "top_k": 2
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8080/predictions/reranker",
    json={
        "query": "летнее платье",
        "offers": [
            "Синее летнее платье",
            "Зимнее пальто",
            "Весенняя куртка"
        ],
        "top_k": 2
    }
)

result = response.json()
for offer in result["results"]:
    print(f"Rank {offer['rank']}: {offer['text']} (score: {offer['score']:.4f})")
```

## Development

### Run Tests

```bash
uv sync --group dev
uv run pytest tests/
```

### Local Development (without Docker)

1. Package the model:
   ```bash
   ./scripts/package_model.sh
   ```

2. Start TorchServe:
   ```bash
   torchserve --start \
     --model-store model-store \
     --models reranker=reranker.mar \
     --ts-config config.properties
   ```

3. Stop TorchServe:
   ```bash
   torchserve --stop
   ```
