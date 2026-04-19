import json
import gzip

# --- CONFIGURATION ---
# Change this to match your exact filename
INPUT_FILE = 'kaikki.org-dictionary-English-verbs.jsonl'
OUTPUT_FILE = '../data/all_english_verbs.json'


def extract_and_save_verbs():
    verbs_list = []

    print(f"Reading from {INPUT_FILE}...")

    # Check if file is zipped (.gz) or unzipped (.jsonl)
    if INPUT_FILE.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open

    count = 0

    # Open the input file
    try:
        with open_func(INPUT_FILE, 'rt', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)

                    # 1. Get the word
                    word = data.get('word')
                    if not word:
                        continue

                    # 2. Extract definitions
                    # We collect all definitions from all senses into one list
                    definitions = []
                    if 'senses' in data:
                        for sense in data['senses']:
                            if 'glosses' in sense:
                                definitions.extend(sense['glosses'])

                    # Only add if we actually found definitions
                    if definitions:
                        # Create a clean object
                        verb_entry = {
                            "word": word,
                            "definitions": definitions
                        }
                        verbs_list.append(verb_entry)

                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} entries...")

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'.")
        print("Please make sure the file name in the script matches your downloaded file.")
        return

    print(f"\nExtraction complete. Found {len(verbs_list)} verbs.")
    print(f"Writing to {OUTPUT_FILE}...")

    # 3. Write to the new file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(verbs_list, f_out, indent=2, ensure_ascii=False)

    print("Done! File is ready.")


if __name__ == "__main__":
    extract_and_save_verbs()