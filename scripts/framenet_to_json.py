import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.corpus import framenet as fn
import json


def main():
    framenet_to_json()
    with open('../data/framenet_data.json', 'r') as f:
        data = json.load(f)
        print("Data type:", type(data))
        if len(data) > 0:
            print("First item type:", type(data[0]))
            print("First item content:", data[0])


def framenet_to_json(output_file='framenet_data.json'):
    print("Fetching Frames...")
    all_frames = fn.frames()

    clean_data = []

    for f in all_frames:
        # We explicitly create a DICTIONARY here
        frame_obj = {
            "frame_id": f.ID,
            "frame_name": f.name,
            "definition": f.definition,
            "lexical_units": [lu.name for lu in f.lexUnit.values()]  # storing just names to save space
        }
        clean_data.append(frame_obj)

    print(f"Saving {len(clean_data)} frames to framenet_data.json...")
    with open('../data/framenet_data.json', 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    print("Success! File is ready for embedding.")




# def download_framenet():
#     """Ensure FrameNet data is downloaded."""
#     try:
#         fn.frames()
#     except LookupError:
#         print("Downloading FrameNet data...")
#         nltk.download('framenet_v17')


if __name__ == "__main__":
    main()
