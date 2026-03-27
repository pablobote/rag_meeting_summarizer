# Local RAG Meeting Summarizer

Lightweight local Retrieval-Augmented Generation (RAG) project for meeting transcripts and documents.

## What this project does

- Transcribes audio into text using Whisper.
- Indexes PDF and TXT files into a local Chroma vector database.
- Runs an automatic local meeting summary using an Ollama model.

## Project structure

```
local_rag_example/
├── data/                  # Source documents (.pdf, .txt)
├── src/
│   ├── ingestion.py       # Audio transcription CLI
│   └── rag.py             # Index + retrieval summary CLI
├── tests/                 # Starter tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- Ollama installed and running locally
- A pulled Ollama model (default used here: `mistral`)

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare documents

1. Put source files in `data/`.
2. Supported file types are `.pdf` and `.txt`.

## Optional: transcribe audio into data/

```powershell
python src/ingestion.py path\to\meeting.mp3 --model-size base --data-dir data
```

## Run the RAG summarizer

```powershell
python src/rag.py --data-dir data --persist-dir chroma_db --llm-model mistral
```
This command builds/updates embeddings and then generates a meeting summary automatically.

By default, the command rebuilds the vector index from files in `data/` so stale embeddings do not affect results.
If you intentionally want to reuse an existing index, add:

```powershell
python src/rag.py --data-dir data --persist-dir chroma_db --llm-model mistral --keep-existing-index
```

## Development

Run tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- `venv/` and `chroma_db/` are intentionally ignored in Git.
- Keep only source code and small sample data in the repository.
