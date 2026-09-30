"""Central configuration for models and chunking runs."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

INDEX_PATH = os.environ.get("INDEX_PATH", "data/vectorstore")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
TOP_K = 5
DEFAULT_CHUNKER_RUN = "pubmedbert_section_450"

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# "NeuML/pubmedbert-base-embeddings"

# Full set of (model, chunker, chunk_size, chunk_overlap) combinations to
# evaluate.
CHUNKER_RUNS = [
    {
        "name": "minilm_section_220",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunker": "section_aware",
        "chunk_size": 220,
        "chunk_overlap": 30,
    },
    {
        "name": "minilm_recursive_220",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunker": "recursive",
        "chunk_size": 220,
        "chunk_overlap": 30,
    },
    {
        "name": "pubmedbert_section_450",
        "model": "NeuML/pubmedbert-base-embeddings",
        "chunker": "section_aware",
        "chunk_size": 450,
        "chunk_overlap": 60,
    },
    {
        "name": "pubmedbert_recursive_450",
        "model": "NeuML/pubmedbert-base-embeddings",
        "chunker": "recursive",
        "chunk_size": 450,
        "chunk_overlap": 60,
    },
]

EMBEDDING_MODEL_NAMES = sorted({run["model"] for run in CHUNKER_RUNS})
