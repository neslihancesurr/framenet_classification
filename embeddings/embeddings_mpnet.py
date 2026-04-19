import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


def generate_embeddings(input_file='data/framenet_data.json', output_file='embeddings/framenet_embedded_mpnet_norm.json'):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading model 'all-mpnet-base-v2' on {device}...")
    model = SentenceTransformer('all-mpnet-base-v2', device=device)

    # 3. Extract Definitions
    # We collect all definitions into a single list to use batch processing
    # This is much faster than calling model.encode() inside the loop

    definitions = [frame['definition'] for frame in frames]
    print(f"Found {len(definitions)} definitions to embed.")
    # 4. Generate Embeddings
    print("Generating embeddings (this may take a while)...")
    # batch_size=32 is standard; increase to 64/128 if you have a powerful GPU
    embeddings = model.encode(
        definitions,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True)

    # 5. Merge Embeddings back into Data
    print("Merging embeddings into JSON structure...")
    for i, frame in enumerate(frames):
        # We must convert the numpy array to a standard Python list for JSON serialization
        frame['embedding'] = embeddings[i].tolist()

        # 6. Save to File
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(frames, f, indent=2, ensure_ascii=False)

        print("Done!")


if __name__ == "__main__":
    generate_embeddings()