import json
import torch
from sentence_transformers import SentenceTransformer, util


def load_embeddings(file_path='embeddings/framenet_embedded_mpnet_norm.json'):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    embeddings = [f['embedding'] for f in frames]
    # Move corpus embeddings to the appropriate device later if needed,
    # but CPU is usually fine for the corpus if it fits in RAM.
    corpus_embeddings = torch.tensor(embeddings)

    return frames, corpus_embeddings


def load_translations(file_path='translation/translated_annotated.json'):
    print(f"Loading data from {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        queries = json.load(file)

    word_array = []
    for query in queries:
        # Using .get() with an empty string default prevents KeyError if a key is missing
        joined = f"{query.get('synset_english', '')}, {query.get('definition_english', '')}"
        word_array.append(joined)

    # Return the full list of dicts AND the text array so we can modify the dicts later
    return queries, word_array


def search_loop():
    frames, corpus_embeddings = load_embeddings()
    queries, word_array = load_translations()

    model_name = 'all-mpnet-base-v2'
    print(f"Loading model {model_name}...")

    # Tap into hardware acceleration for Apple Silicon (M-series chips)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    print("\n" + "=" * 50)
    print("FrameNet Semantic Search Starting")

    # 1. Native Batching
    print(f"Encoding {len(word_array)} queries...")
    # SentenceTransformer handles batching under the hood. batch_size=32 is standard.
    query_embeddings = model.encode(
        word_array,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 2. Vectorized Search
    print("Performing semantic search...")
    # semantic_search processes the entire matrix of queries against the corpus at once
    hits_list = util.semantic_search(query_embeddings, corpus_embeddings, top_k=10)

    # 3. In-Memory Updates
    print("Formatting results...")
    for query_idx, hits in enumerate(hits_list):
        frames_dict = {}
        for rank, hit in enumerate(hits):
            frame_idx = hit['corpus_id']
            frame_name = frames[frame_idx]['frame_name']
            frames_dict[f"frame{rank + 1}"] = frame_name

        # Append the new dictionary to the original query object
        queries[query_idx]['frames'] = frames_dict

    # 4. Single File Write
    output_path = 'data/annotated_frames_mpnet10.json'
    print(f"Writing updated results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(queries, file, indent=4, ensure_ascii=False)

    print("Processing complete!")


if __name__ == "__main__":
    search_loop()