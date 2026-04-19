import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


def generate_embeddings(
    input_file='data/framenet_data.json',
    output_file='embeddings/framenet_embedded_minilm_norm.json'
):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    print(f"Loading model 'all-MiniLM-L6-v2' on {device}...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    # Extract definitions
    definitions = [frame['definition'] for frame in frames]
    print(f"Found {len(definitions)} definitions to embed.")

    # Generate embeddings
    print("Generating embeddings (this may take a while)...")

    embeddings = model.encode(
        definitions,
        batch_size=64,  # MiniLM is lightweight → can increase batch size
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # 🔥 important for cosine similarity
    )

    # Merge embeddings back
    print("Merging embeddings into JSON structure...")
    for i, frame in enumerate(frames):
        frame['embedding'] = embeddings[i].tolist()

    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(frames, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    generate_embeddings()