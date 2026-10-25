# FrameNet Semantic Frame Classification

This project explores matching Turkish synsets and definitions to English FrameNet frames using sentence embeddings. It embeds FrameNet frame definitions with Sentence Transformers, embeds translated query text, and ranks the closest frames by semantic similarity.

The repository includes two embedding models:

- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`)
- **MPNet** (`all-mpnet-base-v2`)

The prediction scripts return the top 10 candidate frames for each input record. `prediction_terminal.py` provides an interactive search that returns the top 5 candidates for a phrase typed at the prompt.

## Repository layout

```text
.
├── data/                 # FrameNet data, source synsets, and prediction outputs
├── embeddings/           # FrameNet embedding files and embedding-generation scripts
├── scripts/              # Data conversion, cleanup, and comparison utilities
├── translation/          # Translation inputs, outputs, and translation scripts
├── prediction_file_minilm.py
├── prediction_file_mpnet.py
├── prediction_terminal.py
└── requirements.txt
```

## Requirements

Python 3.10 or newer is recommended. Install the listed packages in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download the selected Sentence Transformer model. PyTorch will use CUDA, Apple Metal (MPS), or CPU depending on availability in the embedding-generation and file-prediction scripts. The terminal script uses the default device selected by Sentence Transformers/PyTorch.

## Run semantic search over a file

Run commands from the repository root so the relative data paths resolve:

```bash
python prediction_file_mpnet.py
```

This reads:

- `embeddings/framenet_embedded_mpnet_norm.json`
- `translation/translated_annotated.json`

It writes ranked candidates to `data/annotated_frames_mpnet10.json`, in a `frames` field on each query record.

To use MiniLM instead:

```bash
python prediction_file_minilm.py
```

This reads:

- `embeddings/framenet_embedded_minilm_norm.json`
- `data/annotated_frames_mpnet10.json`

It writes results to `data/annotated_frames_minilm10.json`, in a `frames_minilm` field. The MiniLM script's input file is currently the MPNet-named output, so ensure that file contains the intended translated query records before running it.

Each output record retains its original fields and includes up to ten ranked frame names (`frame1` through `frame10`).

## Interactive search

```bash
python prediction_terminal.py
```

Enter an English phrase or sentence to find the five closest FrameNet frames. Type `exit` or `quit` to stop.

The script currently expects its frame embeddings at `data/framenet_embedded_mpnet_norm.json`. The supplied project listing shows the MPNet embedding file under `embeddings/`, so either place/copy the expected file under `data/` or update the default path in `prediction_terminal.py` to `embeddings/framenet_embedded_mpnet_norm.json`.

## Rebuild FrameNet embeddings

If the required FrameNet source and NLTK FrameNet corpus are available, generate frame data and then embeddings:

```bash
python scripts/framenet_to_json.py
python embeddings/embeddings_mpnet.py
# Or, for MiniLM:
python embeddings/embeddings_miniLM.py
```

The embedding scripts expect `data/framenet_data.json` and write normalized embeddings under `embeddings/`. The FrameNet extraction script uses NLTK's FrameNet corpus; install/download that corpus in your Python environment before running it if it is not already present.

## Data preparation and translation utilities

The `scripts/` and `translation/` directories contain additional utilities for cleaning CSV data, extracting English verb entries, finding missing translation records, and translating FrameNet frames or Turkish synsets/definitions. These are supporting scripts rather than a single automated pipeline; inspect each script's configured input/output paths before running it.

Some translation scripts contain placeholder API keys or endpoint values. Configure valid credentials and provider endpoints locally before use, and avoid committing secrets. Translation scripts may send source text to the configured external provider. The project also includes an Ollama-based option intended for a locally hosted model.

## Input and output formats

Frame embedding files are JSON arrays of frame objects. Each object needs at least a `frame_name` and `embedding`; the interactive search also displays `definition`.

The file-prediction scripts expect a JSON array of query objects with English text in `synset_english` and `definition_english`. They combine those fields as the semantic-search query. Missing fields are treated as empty strings.

## Notes

- Similarity search produces ranked candidates; it does not make a final frame annotation or evaluate accuracy against gold labels.
- Embeddings and input/output paths are configured in the scripts. Run from the repository root unless you have changed those paths.
- The repository's `requirements.txt` pins a broad environment, including packages used by auxiliary scripts. If installation fails on a particular platform or Python version, the dependency list may need to be adjusted for that environment.
