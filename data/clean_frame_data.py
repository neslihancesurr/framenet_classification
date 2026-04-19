import json
import re

def extract_examples_and_clean(definition):
    # Extract all non-empty quoted examples
    examples = re.findall(r"'([^']+)'", definition)

    # Remove quoted parts from definition
    clean_definition = re.sub(r"\s*'[^']*'", "", definition).strip()

    # Optional: normalize extra spaces
    clean_definition = re.sub(r"\s{2,}", " ", clean_definition)

    return clean_definition, examples


def process_frames(input_file, output_file):
    # Load JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each entry
    for entry in data:
        definition = entry.get("definition", "")

        clean_def, examples = extract_examples_and_clean(definition)

        # Update fields
        entry["definition"] = clean_def
        entry["examples"] = examples

    # Save updated JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Example usage
process_frames("framenet_data.json", "framenet_data_cleaned.json")