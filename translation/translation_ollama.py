import pandas as pd
import json
import time
import os
import requests

# 1. Setup
OLLAMA_URL = "local host URL"
MODEL_ID = "gemma3"
INPUT_FILE = '../data/Synsets_Definitions.csv'
OUTPUT_FILE = 'discarded/translated_results_ollama.json'
BATCH_SIZE = 30
BASE_DELAY = 2  # Ollama is local, no rate limits needed

RESPONSE_SCHEMA_DESCRIPTION = """
Return a JSON array where each object has exactly these fields:
- synset_turkish: string (original Turkish synset)
- definition_turkish: string (original Turkish definition)
- synset_english: string (English translation of synset)
- definition_english: string (English translation of definition)

Example format:
[
  {
    "synset_turkish": "...",
    "definition_turkish": "...",
    "synset_english": "...",
    "definition_english": "..."
  }
]
"""


def extract_json_from_response(text):
    """Extract JSON array from model response, handling markdown code blocks."""
    # Try to find JSON array in the response
    text = text.strip()

    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find the first '[' and last ']'
    start = text.find('[')
    end = text.rfind(']')

    if start != -1 and end != -1:
        json_str = text[start:end + 1]
        return json.loads(json_str)

    raise ValueError("No valid JSON array found in response")


def translate_batch(batch_df, retries=5):
    input_strings = []
    for _, row in batch_df.iterrows():
        s = row.get('SYNSET', row.get('synset', ''))
        d = row.get('DEFINITION', row.get('definition', ''))
        input_strings.append(f"synset:{s},definition:{d}")

    input_payload = "\n".join(input_strings)

    prompt = f"""You are an expert linguist. Translate the following Turkish synsets and definitions into English.

Input data:
{input_payload}

Maintain the original Turkish values and provide accurate English translations.

{RESPONSE_SCHEMA_DESCRIPTION}

Return ONLY the JSON array, no explanation or extra text."""

    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4096
                    }
                },
                timeout=300  # Local models can be slow on large batches
            )
            response.raise_for_status()
            result_text = response.json().get("response", "")
            return extract_json_from_response(result_text)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None


# 2. Checkpoint Logic
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    processed_count = len(all_results)
    print(f"Resuming from row {processed_count}...")
else:
    all_results = []
    processed_count = 0  # Start from beginning since there's no rate limit

# 3. Processing
df = pd.read_csv(INPUT_FILE)

for i in range(processed_count, len(df), BATCH_SIZE):
    print(f"Processing rows {i} to {min(i + BATCH_SIZE, len(df))}...")
    batch = df.iloc[i: i + BATCH_SIZE]

    translated_data = translate_batch(batch)

    if translated_data:
        all_results.extend(translated_data)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"  ✓ Saved {len(translated_data)} entries (total: {len(all_results)})")
    else:
        print(f"  ✗ Batch failed, skipping rows {i}-{min(i + BATCH_SIZE, len(df))}")

    time.sleep(BASE_DELAY)  # Small delay to avoid overwhelming local Ollama

print(f"Finished! Processed {len(all_results)} entries.")