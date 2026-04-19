"""
Find rows in the CSV that are missing from the JSON file.
Missing means: no JSON entry whose synset_turkish matches the CSV's SYNSET.

Usage:
    python find_missing.py <csv_file> <json_file> [output_csv]

Example:
    python find_missing.py synsets.csv translations.json missing.csv
"""

import sys
import csv
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python find_missing.py <csv_file> <json_file> [output_csv]")
        sys.exit(1)

    csv_path   = sys.argv[1]
    json_path  = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "missing.csv"

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(row)

    print(f"CSV rows loaded : {len(csv_rows)}")

    # ── Load JSON ─────────────────────────────────────────────────────────────
    with open(json_path, encoding="utf-8") as f:
        json_entries = json.load(f)

    print(f"JSON entries    : {len(json_entries)}")

    # Build a set of synset_turkish values present in the JSON.
    # Strip whitespace so minor spacing differences don't cause false misses.
    json_synsets = {entry["definition_turkish"].strip() for entry in json_entries}

    # ── Find missing rows ─────────────────────────────────────────────────────
    missing = [
        row for row in csv_rows
        if row["DEFINITION"].strip() not in json_synsets
    ]

    print(f"Missing entries : {len(missing)}")

    # ── Write output CSV ──────────────────────────────────────────────────────
    if missing:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["SYNSET", "DEFINITION"])
            writer.writeheader()
            writer.writerows(missing)
        print(f"Written to      : {output_path}")
    else:
        print("No missing entries — files are in sync!")

if __name__ == "__main__":
    main()