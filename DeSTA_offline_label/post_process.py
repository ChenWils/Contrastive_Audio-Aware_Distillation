#!/usr/bin/env python3
"""
Post-process JSONL file to transform format for audio conversation data.

Input format:
{
  "audio_filepath": "...",
  "transcription": "...",
  "messages": [system, user, assistant],
  "target": "...",
  ...
}

Output format:
{
  "audios": [{"audio_filepath": "...", "transcription": "..."}],
  "messages": [system, user],  # only system and user, no assistant
  "target": "..."
}
"""

import json
import argparse
from pathlib import Path


def transform_entry(entry):
    """Transform a single entry from input format to output format."""
    # Extract audios information
    audios = [{
        "audio_filepath": entry["audio_filepath"],
        "transcription": entry["transcription"]
    }]

    # Filter messages to only include system and user roles (exclude assistant)
    messages = [
        msg for msg in entry["messages"]
        if msg["role"] in ["system", "user"]
    ]

    # Create output entry
    output_entry = {
        "audios": audios,
        "messages": messages,
        "target": entry["target"]
    }

    return output_entry


def post_process_jsonl(input_file, output_file=None):
    """
    Post-process JSONL file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (optional, defaults to input_file with _processed suffix)
    """
    input_path = Path(input_file)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_processed.jsonl"
    else:
        output_file = Path(output_file)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_file}")

    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                transformed = transform_entry(entry)
                outfile.write(json.dumps(transformed, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    print(f"\nProcessed {processed_count} entries successfully!")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process audio conversation JSONL files"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: input_file with _processed suffix)"
    )

    args = parser.parse_args()
    post_process_jsonl(args.input_file, args.output)


if __name__ == "__main__":
    main()
