"""Example client for testing the reranker service."""

import msgpack
import requests

# Example request data
request_data = {
    "query": "летнее платье",
    "offers": [
        "Синее летнее платье из хлопка",
        "Зимнее пальто с мехом",
        "Легкое платье с цветочным принтом",
        "Теплая куртка на пуху",
        "Короткое платье для лета",
    ],
    "top_k": 3,
}

# Send request to the service
response = requests.post(
    "http://localhost:8000/rerank",
    data=msgpack.packb(request_data),
    headers={"Content-Type": "application/msgpack"},
)

# Parse response
result = msgpack.unpackb(response.content)

# Display results
print(f"Query: {request_data['query']}")
print(f"\nTop {request_data['top_k']} results:")
print("-" * 80)

for offer in result["results"]:
    print(f"Rank {offer['rank']}: {offer['text']}")
    print(f"  Score: {offer['score']:.4f}")
    print()

