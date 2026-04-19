import pandas as pd
import json
import time
import os
import requests

# ─────────────────────────────────────────
# 1. Configuration — change these as needed
# ─────────────────────────────────────────

PROVIDER = "openrouter"  # "ollama" or "openrouter"

OLLAMA_CONFIG = {
    "url": "local host url",
    "model": "gemma3",
}

OPENROUTER_CONFIG = {
    "url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": "API Key",
    "model": "google/gemma-4-26b-a4b-it:free",  # Any model slug from openrouter.ai/models
}

INPUT_FILE = '../data/Synsets_Definitions.csv'
OUTPUT_FILE = 'discarded/translated_results_gemma4.json'
BATCH_SIZE = 30
BASE_DELAY = 10  # Increase if hitting OpenRouter rate limits


# ─────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────

SCHEMA_DESCRIPTION = """
Return a JSON array where each object has exactly these fields:
- synset_turkish: string (original Turkish synset)
- definition_turkish: string (original Turkish definition)
- synset_english: string (English translation of synset)
- definition_english: string (English translation of definition)

Example:
[
  {
    "synset_turkish": "...",
    "definition_turkish": "...",
    "synset_english": "...",
    "definition_english": "..."
  }
]

Return ONLY the JSON array, no explanation or extra text."""


def extract_json_from_response(text):
    """Extract JSON array from model response, handling markdown code blocks."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise ValueError("No valid JSON array found in response")


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_CONFIG["url"],
        json={
            "model": OLLAMA_CONFIG["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096
            }
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json().get("response", "")


def call_openrouter(prompt):
    response = requests.post(
        OPENROUTER_CONFIG["url"],
        headers={
            "Authorization": f"Bearer {OPENROUTER_CONFIG['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.com",   # Optional but recommended by OpenRouter
            "X-Title": "Synset Translator",            # Shows up in your OpenRouter dashboard
        },
        json={
            "model": OPENROUTER_CONFIG["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def translate_batch(batch_df, retries=5):
    input_strings = []
    for _, row in batch_df.iterrows():
        s = row.get('SYNSET', row.get('synset', ''))
        d = row.get('DEFINITION', row.get('definition', ''))
        input_strings.append(f"synset:{s},definition:{d}")

    prompt = f"""You are an expert linguist. Translate the following Turkish synsets and definitions into English.

Input data:
{chr(10).join(input_strings)}

Maintain the original Turkish values and provide accurate English translations.

{SCHEMA_DESCRIPTION}"""

    for attempt in range(retries):
        try:
            if PROVIDER == "ollama":
                raw = call_ollama(prompt)
            elif PROVIDER == "openrouter":
                raw = call_openrouter(prompt)
            else:
                raise ValueError(f"Unknown provider: '{PROVIDER}'. Use 'ollama' or 'openrouter'.")

            return extract_json_from_response(raw)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"  HTTP {status} error (attempt {attempt + 1}): {e}")
            if status == 429:
                wait = (2 ** attempt) + 10
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif status in (500, 502, 503):
                time.sleep(5)
                continue
            else:
                print("  Unrecoverable HTTP error, skipping batch.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")

        time.sleep(2)

    return None


# ─────────────────────────────────────────
# 3. Checkpoint
# ─────────────────────────────────────────

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    processed_count = len(all_results)
    print(f"Resuming from row {processed_count}...")
else:
    all_results = []
    processed_count = 0

# ─────────────────────────────────────────
# 4. Main loop
# ─────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
provider_label = f"{PROVIDER} / {OLLAMA_CONFIG['model'] if PROVIDER == 'ollama' else OPENROUTER_CONFIG['model']}"
print(f"Provider: {provider_label} | Rows: {len(df)} | Batch size: {BATCH_SIZE}")

for i in range(processed_count, len(df), BATCH_SIZE):
    end_row = min(i + BATCH_SIZE, len(df))
    print(f"Processing rows {i}–{end_row}...")

    batch = df.iloc[i:end_row]
    translated_data = translate_batch(batch)

    if translated_data:
        all_results.extend(translated_data)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"  ✓ {len(translated_data)} entries saved (total: {len(all_results)})")
    else:
        print(f"  ✗ Batch failed, skipping rows {i}–{end_row}")

    time.sleep(BASE_DELAY)

print(f"\nDone! {len(all_results)} total entries in {OUTPUT_FILE}")