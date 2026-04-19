import json
import torch
from sentence_transformers import SentenceTransformer, util


def main():
    search_loop()


def load_data(file_path='data/framenet_embedded_mpnet_norm.json'):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    embeddings = [f['embedding'] for f in frames]
    corpus_embeddings = torch.tensor(embeddings)

    return frames, corpus_embeddings


def search_loop():
    frames, corpus_embeddings = load_data()

    model_name = 'all-mpnet-base-v2'
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)

    print("\n" + "=" * 50)
    print("FrameNet Semantic Search Ready!")
    print("Type a sentence or phrase to find the closest Frame.")
    print("Type 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        query = input("Query: ")
        if query.lower() in ['exit', 'quit']:
            break

        query_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)

        for hit in hits[0]:
            frame_idx = hit['corpus_id']
            score = hit['score']
            frame = frames[frame_idx]

            print(f"  [{score:.4f}] Frame: {frame['frame_name']}")
            # Truncate definition for cleaner output
            short_def = (frame['definition'][:150] + '...') if len(frame['definition']) > 150 else frame['definition']
            print(f"           Def: {short_def}")
            print("-" * 40)
        print("\n")


if __name__ == "__main__":
    main()