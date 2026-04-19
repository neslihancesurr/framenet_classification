import json
import torch
from sentence_transformers import SentenceTransformer, util


def load_embeddings(file_path='embeddings/framenet_embedded_minilm_norm.json'):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    embeddings = [f['embedding'] for f in frames]

    corpus_embeddings = torch.tensor(embeddings)

    # Safety normalization (in case something was missed during saving)
    #corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)

    return frames, corpus_embeddings


def load_translations(file_path="data/annotated_frames_mpnet10.json"):
    print(f"Loading data from {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        queries = json.load(file)

    word_array = []
    for query in queries:
        joined = f"{query.get('synset_english', '')}, {query.get('definition_english', '')}"
        word_array.append(joined)

    return queries, word_array


def search_loop():
    frames, corpus_embeddings = load_embeddings()
    queries, word_array = load_translations()

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Loading model {model_name}...")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    print("\n" + "=" * 50)
    print("FrameNet Semantic Search (MiniLM) Starting")

    # 1. Encode queries
    print(f"Encoding {len(word_array)} queries...")
    query_embeddings = model.encode(
        word_array,
        batch_size=64,  # MiniLM → can use larger batch
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True  # 🔥 critical for fair comparison
    )

    # 2. Semantic search
    print("Performing semantic search...")
    hits_list = util.semantic_search(query_embeddings, corpus_embeddings, top_k=10)

    # 3. Format results
    print("Formatting results...")
    for query_idx, hits in enumerate(hits_list):
        frames_dict = {}
        for rank, hit in enumerate(hits):
            frame_idx = hit['corpus_id']
            frame_name = frames[frame_idx]['frame_name']
            frames_dict[f"frame{rank + 1}"] = frame_name

        queries[query_idx]['frames_minilm'] = frames_dict  # 🔥 keep separate from MPNet

    # 4. Save results
    output_path = 'data/annotated_frames_minilm10.json'
    print(f"Writing updated results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(queries, file, indent=4, ensure_ascii=False)

    print("Processing complete!")


if __name__ == "__main__":
    search_loop()