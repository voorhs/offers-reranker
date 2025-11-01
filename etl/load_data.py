import json
import os
from pathlib import Path

from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient

load_dotenv()


async def main():
    query_template = """
    SELECT attributes
    FROM records
    WHERE span_name LIKE 'Chat Completion with %'
    ORDER BY created_at
    LIMIT {batch_size} OFFSET {offset}
    """

    batch_size = 10000
    offset = 0
    total_fetched = 0
    total_samples = 0

    # Open JSONL file for writing incrementally
    with Path("samples.jsonl").open("w", encoding="utf-8") as output_file:
        async with AsyncLogfireQueryClient(
            read_token=os.getenv("LOGFIRE_READ_TOKEN"),
            timeout=10,
        ) as client:
            while True:
                print(f"Fetching batch at offset {offset}...")
                query = query_template.format(offset=offset, batch_size=batch_size)

                # Load data as JSON, in row-oriented format
                json_rows = await client.query_json_rows(sql=query, limit=batch_size)

                batch_rows = json_rows["rows"]
                if not batch_rows:
                    print(f"No more rows to fetch. Total rows fetched: {total_fetched}")
                    break

                total_fetched += len(batch_rows)

                # Process and write samples on the fly to save memory
                batch_samples = 0
                for row in batch_rows:
                    messages = row["attributes"]["request_data"]["messages"]
                    if not messages[0]["content"].startswith("1. \nClient searched"):
                        continue
                    if not row["attributes"].get("response_data"):
                        continue

                    # Extract training sample immediately
                    request = row["attributes"]["request_data"]["messages"][1]["content"][0]["text"].strip()
                    response = row["attributes"]["response_data"]["message"]["tool_calls"][0]["function"]["arguments"]
                    sample = {"request": request, "response": response}

                    # Write to JSONL file immediately
                    output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    batch_samples += 1
                    total_samples += 1

                print(
                    f"Fetched {len(batch_rows)} rows, wrote {batch_samples} samples. Total: {total_fetched} fetched, {total_samples} samples written"
                )

                # If we got fewer than batch_size rows, we've reached the end
                if len(batch_rows) < batch_size:
                    print(f"Reached end of results. Total: {total_fetched} fetched, {total_samples} samples written")
                    break

                offset += batch_size

    print("\nAll samples written to samples.jsonl")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
