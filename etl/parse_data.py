"""Script to parse raw samples.jsonl data into structured format for ranking model."""

import json
import re
from pathlib import Path


def parse_sample(line: str) -> dict:
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


def parse_all_samples(input_path: str, output_path: str) -> None:
    """Parse all samples from input JSONL file and save to output.

    Args:
        input_path: Path to input samples.jsonl file
        output_path: Path to output parsed data file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    parsed_samples = []
    errors = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parsed = parse_sample(line)
                parsed_samples.append(parsed)
            except Exception as e:
                errors.append((line_num, str(e)))
                print(f"Error parsing line {line_num}: {e}")

    # Save parsed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_samples, f, ensure_ascii=False, indent=2)

    print("\nParsing complete!")
    print(f"Successfully parsed: {len(parsed_samples)} samples")
    print(f"Errors: {len(errors)} lines")
    print(f"Output saved to: {output_file}")

    if errors:
        print("\nFirst few errors:")
        for line_num, error in errors[:5]:
            print(f"  Line {line_num}: {error}")


def main():
    """
    Main entry point for parsing raw samples.

    Input format (samples.jsonl):
        Each line is a JSON object with:
        - request: XML-like string with <UserQuery> and <Offer> tags
        - response: JSON string with 'scores' array

    Output format (parsed_samples.json):
        JSON array where each element is:
        {
            "user_query": str,
            "offers": [[offer_text: str, score: int], ...]
        }

    Note: offers are stored as lists [text, score] in JSON (equivalent to tuples)
    """
    input_path = "data/samples.jsonl"
    output_path = "data/parsed_samples.json"

    print(f"Parsing {input_path}...")
    parse_all_samples(input_path, output_path)

    # Show example of parsed data
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data:
        print("\n" + "=" * 80)
        print("Example parsed sample:")
        print("=" * 80)
        sample = data[0]
        print(f"\nUser Query:\n{sample['user_query']}\n")
        print(f"Number of offers: {len(sample['offers'])}\n")
        print("First 3 offers:")
        for i, (offer_text, score) in enumerate(sample["offers"][:3], 1):
            print(f"\n--- Offer {i} (Score: {score}) ---")
            print(offer_text[:200] + "..." if len(offer_text) > 200 else offer_text)


if __name__ == "__main__":
    main()
