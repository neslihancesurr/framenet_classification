import pandas as pd
from google import genai
from google.genai import types
import json
import time
import os

# First 170 rows in CSV with gemini-3-flash-preview. It worked really well.
# 150-160 arası alınmamış, 160-170 arası yapılmış, sonrası yok (translated results)
# Sonraki 830'dan başlamalı, oraya kadar translated_result2de yapıldı
# failed requestleri de sayıyor - boş request olmasın diye sleep time 65 yaptım
# Try again tomorrow RPD=20(i exceed this, improve batch to send less requests)
# RPM=5 (i almost exceed this, make sleep time longer)
# TPM I'm okay with this
# 2 API key ve 50lik batch size ile günde 2000 row yapabilmem lazım

# flash 2.5 ve prolar free-tier çalışmıyor.
# "gemini-3.1-flash-lite-preview"
# "gemini-3-flash-preview"


# 1. Setup
client = genai.Client(api_key="API-key")
MODEL_ID = "gemini-3.1-flash-lite-preview"
INPUT_FILE = 'missing.csv'
OUTPUT_FILE = 'translated_annotated_missing.json'
BATCH_SIZE = 20
BASE_DELAY = 12

# Define the exact JSON structure we want
RESPONSE_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'synset_turkish': {'type': 'STRING'},
            'definition_turkish': {'type': 'STRING'},
            'synset_english': {'type': 'STRING'},
            'definition_english': {'type': 'STRING'}
        },
        'required': ['synset_turkish', 'definition_turkish', 'synset_english', 'definition_english']
    }
}


def translate_batch(batch_df, retries=5):
    # Format input exactly as requested: "synset:<content>,definition:<content>"
    input_strings = []
    for _, row in batch_df.iterrows():
        # Using .get() to handle case sensitivity in CSV headers (SYNSET vs synset)
        s = row.get('SYNSET', row.get('synset', ''))
        d = row.get('DEFINITION', row.get('definition', ''))
        input_strings.append(f"synset:{s},definition:{d}")

    input_payload = "\n".join(input_strings)

    prompt = f"""
    You are an expert linguist. Translate the following Turkish synsets and definitions into English.

    Input data:
    {input_payload}

    Maintain the original Turkish values and provide accurate English translations.
    """

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=RESPONSE_SCHEMA,  # Force the exact JSON structure
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                # Backoff: wait 6s, 8s, 12s, 20s...
                wait_time = (2 ** attempt) + 10
                print(f"Server busy. Retrying in {wait_time}s... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                print(f"Skipping batch due to error: {e}")
                return None
    return None


# 2. Checkpoint Logic
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    processed_count = len(all_results)
    print(f"Resuming from row {processed_count}...")
else:
    all_results = []
    processed_count = 0

# 3. Processing
df = pd.read_csv(INPUT_FILE)
batch_size = BATCH_SIZE

for i in range(processed_count, len(df), batch_size):
    print(f"Processing rows {i} to {min(i + batch_size, len(df))}...")
    batch = df.iloc[i: i + batch_size]

    translated_data = translate_batch(batch)

    if translated_data:
        all_results.extend(translated_data)
        # Save after every batch to keep progress safe
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    time.sleep(50)  # Respect the Free Tier RPM

print(f"Finished! Processed {len(all_results)} entries.")