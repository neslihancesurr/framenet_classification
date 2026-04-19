import pandas as pd
from google import genai
from google.genai import types
import json
import time
import os

# 1. Setup
# IMPORTANT: Revoke the old key and load the new one from environment variables!
client = genai.Client(api_key="API KEY")
MODEL_ID = "gemini-3-flash-preview"
INPUT_FILE = '../data/framenet_data.json'
OUTPUT_FILE = 'discarded/translated_frames_gemini3.json'
BATCH_SIZE = 20

# Corrected Schema Dictionary
RESPONSE_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'frame_id': {'type': 'INTEGER'},
            'frame_name': {'type': 'STRING'},
            'definition': {'type': 'STRING'},
            'lexical_units': {  # Fixed naming and array syntax
                'type': 'ARRAY',
                'items': {'type': 'STRING'}
            }
        },
        'required': ['frame_id', 'frame_name', 'definition', 'lexical_units']
    }
}


def translate_batch(batch_df, retries=5):
    input_strings = []

    # Fill NaN values with empty strings to prevent payload formatting errors
    batch_df = batch_df.fillna('')

    for _, row in batch_df.iterrows():
        # Crucial: Extract the ID so the model knows what to return
        frame_id = row.get('frame_id', 'UNKNOWN_ID')
        frame_name = row.get('frame_name', '')
        definition = row.get('definition', '')
        lexical_unit = row.get('lexical_units', '')

        # Include the ID in the payload sent to the model
        input_strings.append(
            f"frame_id:{frame_id} | frame name:{frame_name} | definition:{definition} | lexical units:{lexical_unit}")

    input_payload = "\n".join(input_strings)

    prompt = f"""
    You are an expert linguist. Translate the following English frame names, frame definitions, and lexical units into Turkish.

    Input data:
    {input_payload}

    Maintain the exact same frame_id provided in the input data. Provide accurate Turkish translations for the other fields.
    """

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                wait_time = (2 ** attempt) + 10
                print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{retries})")
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
try:
    df = pd.read_json(INPUT_FILE)
except ValueError as e:
    print(f"Error reading input JSON. Ensure it is a valid list of dictionaries. {e}")
    exit()

for i in range(processed_count, len(df), BATCH_SIZE):
    print(f"Processing rows {i} to {min(i + BATCH_SIZE, len(df))}...")
    batch = df.iloc[i: i + BATCH_SIZE]

    translated_data = translate_batch(batch)

    if translated_data:
        all_results.extend(translated_data)

        # Save atomically using a temporary file to prevent corruption if script is killed
        temp_output = OUTPUT_FILE + ".tmp"
        with open(temp_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        os.replace(temp_output, OUTPUT_FILE)

    # 40s sleep + execution time keeps you safely under the 15 RPM Free Tier limit
    time.sleep(40)

print(f"Finished! Processed {len(all_results)} entries.")