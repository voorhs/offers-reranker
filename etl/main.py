"""Combined ETL pipeline for offers reranker data.

This script combines data loading from Logfire, parsing, and splitting into train/test/dev sets.
"""

import json
import os
import re
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger

load_dotenv()

app = cyclopts.App(
    name="pipeline",
    help="ETL pipeline for offers reranker: load, parse, and split data",
)


def parse_sample(line: str) -> dict[str, Any]:
    """Parse a single JSONL line into structured format.

    Args:
        line: JSON string containing request and response fields

    Returns:
        dict with keys:
            - user_query: str - the user's search query
            - offers: list[tuple[str, int]] - list of (offer_text, score) tuples
    """
    # Load JSON
    data = json.loads(line)
    request_text = data["request"]
    response_json = json.loads(data["response"])
    scores = response_json["scores"]

    # Extract user query using regex
    user_query_match = re.search(r"<UserQuery>(.*?)</UserQuery>", request_text, re.DOTALL)
    user_query = user_query_match.group(1).strip() if user_query_match else ""

    # Extract offers using regex
    # Pattern matches: <Offer i=X>content</Offer>
    offer_pattern = r"<Offer i=\d+>(.*?)</Offer>"
    offer_matches = re.findall(offer_pattern, request_text, re.DOTALL)

    offers = []
    for i, offer_text in enumerate(offer_matches):
        offer_text = offer_text.strip()
        # Get corresponding score from response
        score = scores[i] if i < len(scores) else 0
        offers.append((offer_text, score))

    return {"user_query": user_query, "offers": offers}


@app.command
async def load(
    output_path: Annotated[
        Path,
        cyclopts.Parameter(help="Path to output JSONL file"),
    ] = Path("data/samples.jsonl"),
    batch_size: Annotated[
        int,
        cyclopts.Parameter(help="Number of records to fetch per batch"),
    ] = 10000,
    timeout: Annotated[
        int,
        cyclopts.Parameter(help="Query timeout in seconds"),
    ] = 10,
) -> None:
    """Load raw data from Logfire and save to JSONL file."""
    logger.info(f"Starting data load to {output_path}")

    query_template = """
    SELECT attributes
    FROM records
    WHERE span_name LIKE 'Chat Completion with %'
    ORDER BY created_at
    LIMIT {batch_size} OFFSET {offset}
    """

    offset = 0
    total_fetched = 0
    total_samples = 0

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open JSONL file for writing incrementally
    with output_path.open("w", encoding="utf-8") as output_file:
        async with AsyncLogfireQueryClient(
            read_token=os.getenv("LOGFIRE_READ_TOKEN") or "fake",
            timeout=timeout,  # type: ignore[arg-type]
        ) as client:
            while True:
                logger.info(f"Fetching batch at offset {offset}...")
                query = query_template.format(offset=offset, batch_size=batch_size)

                # Load data as JSON, in row-oriented format
                json_rows = await client.query_json_rows(sql=query, limit=batch_size)

                batch_rows = json_rows["rows"]
                if not batch_rows:
                    logger.info(f"No more rows to fetch. Total rows fetched: {total_fetched}")
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

                logger.info(
                    f"Fetched {len(batch_rows)} rows, wrote {batch_samples} samples. "
                    f"Total: {total_fetched} fetched, {total_samples} samples written"
                )

                # If we got fewer than batch_size rows, we've reached the end
                if len(batch_rows) < batch_size:
                    logger.info(f"Reached end of results. Total: {total_fetched} fetched, {total_samples} samples")
                    break

                offset += batch_size

    logger.success(f"All samples written to {output_path}")


@app.command
def parse(
    input_path: Annotated[
        Path,
        cyclopts.Parameter(help="Path to input JSONL file"),
    ] = Path("data/samples.jsonl"),
    output_path: Annotated[
        Path,
        cyclopts.Parameter(help="Path to output parsed JSON file"),
    ] = Path("data/parsed_samples.json"),
    show_example: Annotated[
        bool,
        cyclopts.Parameter(help="Show example of parsed data"),
    ] = True,
) -> None:
    """Parse raw JSONL data into structured format."""
    logger.info(f"Starting parsing of {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_samples = []
    errors = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parsed = parse_sample(line)
                parsed_samples.append(parsed)
            except Exception as e:
                errors.append((line_num, str(e)))
                logger.error(f"Error parsing line {line_num}: {e}")

    # Save parsed data
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_samples, f, ensure_ascii=False, indent=2)

    logger.success("Parsing complete!")
    logger.info(f"Successfully parsed: {len(parsed_samples)} samples")
    logger.info(f"Errors: {len(errors)} lines")
    logger.info(f"Output saved to: {output_path}")

    if errors:
        logger.warning("First few errors:")
        for line_num, error in errors[:5]:
            logger.warning(f"  Line {line_num}: {error}")

    # Show example of parsed data
    if show_example and parsed_samples:
        logger.info("=" * 80)
        logger.info("Example parsed sample:")
        logger.info("=" * 80)
        sample = parsed_samples[0]
        logger.info(f"\nUser Query:\n{sample['user_query']}\n")
        logger.info(f"Number of offers: {len(sample['offers'])}\n")
        logger.info("First 3 offers:")
        for i, (offer_text, score) in enumerate(sample["offers"][:3], 1):
            logger.info(f"\n--- Offer {i} (Score: {score}) ---")
            offer_preview = offer_text[:200] + "..." if len(offer_text) > 200 else offer_text
            logger.info(offer_preview)


@app.command
def split(
    input_path: Annotated[
        Path,
        cyclopts.Parameter(help="Path to input parsed JSON file"),
    ] = Path("data/parsed_samples.json"),
    output_dir: Annotated[
        Path,
        cyclopts.Parameter(help="Directory to save split files"),
    ] = Path("data/splits"),
    train_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for training set"),
    ] = 0.7,
    dev_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for development set"),
    ] = 0.15,
    test_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for test set"),
    ] = 0.15,
    seed: Annotated[
        int,
        cyclopts.Parameter(help="Random seed for reproducibility"),
    ] = 42,
) -> None:
    """Split parsed data into train/dev/test sets."""
    import random

    logger.info(f"Starting data split from {input_path}")

    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Load data
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    logger.info(f"Loaded {total_samples} samples")

    # Shuffle data with seed for reproducibility
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split indices
    train_end = int(total_samples * train_ratio)
    dev_end = train_end + int(total_samples * dev_ratio)

    # Split data
    train_data = shuffled_data[:train_end]
    dev_data = shuffled_data[train_end:dev_end]
    test_data = shuffled_data[dev_end:]

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    splits = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data,
    }

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        logger.success(f"Saved {len(split_data)} samples to {output_path}")

    logger.success("Data splitting complete!")
    logger.info(
        f"Train: {len(train_data)} ({len(train_data) / total_samples:.1%}), "
        f"Dev: {len(dev_data)} ({len(dev_data) / total_samples:.1%}), "
        f"Test: {len(test_data)} ({len(test_data) / total_samples:.1%})"
    )


@app.default
async def full_pipeline(
    raw_output: Annotated[
        Path,
        cyclopts.Parameter(help="Path to save raw JSONL file"),
    ] = Path("../data/samples.jsonl"),
    parsed_output: Annotated[
        Path,
        cyclopts.Parameter(help="Path to save parsed JSON file"),
    ] = Path("../data/parsed_samples.json"),
    split_dir: Annotated[
        Path,
        cyclopts.Parameter(help="Directory to save train/dev/test splits"),
    ] = Path("../data/splits"),
    batch_size: Annotated[
        int,
        cyclopts.Parameter(help="Number of records to fetch per batch from Logfire"),
    ] = 10000,
    train_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for training set"),
    ] = 0.7,
    dev_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for development set"),
    ] = 0.15,
    test_ratio: Annotated[
        float,
        cyclopts.Parameter(help="Ratio of data for test set"),
    ] = 0.15,
    seed: Annotated[
        int,
        cyclopts.Parameter(help="Random seed for reproducibility"),
    ] = 42,
    skip_load: Annotated[
        bool,
        cyclopts.Parameter(help="Skip loading data from Logfire (use existing raw file)"),
    ] = False,
    skip_parse: Annotated[
        bool,
        cyclopts.Parameter(help="Skip parsing step (use existing parsed file)"),
    ] = False,
) -> None:
    """Run the full ETL pipeline: load -> parse -> split."""
    logger.info("Starting full ETL pipeline")

    # Step 1: Load data
    if not skip_load:
        logger.info("Step 1: Loading data from Logfire")
        await load(output_path=raw_output, batch_size=batch_size)
    else:
        logger.info(f"Step 1: Skipped (using existing {raw_output})")

    # Step 2: Parse data
    if not skip_parse:
        logger.info("Step 2: Parsing data")
        parse(input_path=raw_output, output_path=parsed_output, show_example=False)
    else:
        logger.info(f"Step 2: Skipped (using existing {parsed_output})")

    # Step 3: Split data
    logger.info("Step 3: Splitting data")
    split(
        input_path=parsed_output,
        output_dir=split_dir,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    logger.success("Full pipeline completed successfully!")


if __name__ == "__main__":
    app()
