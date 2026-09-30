"""A small local cosine-search index backed by NumPy and JSON."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


@dataclass
class VectorIndex:
    chunks: list[Document]
    embeddings: np.ndarray
    model_name: str
    configuration: dict


def _validate(index: VectorIndex) -> None:
    vectors = index.embeddings
    if not index.chunks or vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Index must contain chunks and a nonempty embedding matrix.")
    if len(index.chunks) != len(vectors):
        raise ValueError("Chunk count does not match embedding rows.")
    ids = [chunk.metadata.get("chunk_id") for chunk in index.chunks]
    if not all(isinstance(value, str) and value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Chunks must have unique, nonempty string chunk IDs.")
    if not index.model_name or not np.isfinite(vectors).all():
        raise ValueError("Index needs a model name and finite embeddings.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Embeddings must be L2-normalized.")


def save_index(path, chunks, embeddings, model_name, configuration=None) -> None:
    """Save one model/configuration per directory; rebuilding replaces it."""
    index = VectorIndex(
        list(chunks),
        np.asarray(embeddings, dtype=np.float32),
        model_name,
        configuration or {},
    )
    _validate(index)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "model_name": model_name,
        "configuration": index.configuration,
        "chunks": [
            {"page_content": c.page_content, "metadata": c.metadata} for c in index.chunks
        ],
    }
    # Serialize first, before touching an existing index.
    json.dumps(metadata)
    temporary = path / "embeddings.tmp.npy"
    np.save(temporary, index.embeddings, allow_pickle=False)
    metadata["embeddings_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary_metadata = path / "chunks.tmp.json"
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path / "embeddings.npy")
    temporary_metadata.replace(path / "chunks.json")


def load_index(path) -> VectorIndex:
    path = Path(path)
    if not (path / "chunks.json").exists() or not (path / "embeddings.npy").exists():
        raise FileNotFoundError(f"No complete index at {path}. Run build-index first.")
    metadata = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    if metadata.get("version") != 1:
        raise ValueError("Unsupported index version. Rebuild the index.")
    digest = hashlib.sha256((path / "embeddings.npy").read_bytes()).hexdigest()
    if digest != metadata.get("embeddings_sha256"):
        raise ValueError("Index files do not match. Rebuild the index.")
    index = VectorIndex(
        [Document(**chunk) for chunk in metadata["chunks"]],
        np.load(path / "embeddings.npy", allow_pickle=False),
        metadata["model_name"],
        metadata["configuration"],
    )
    _validate(index)
    return index


def search(
    index: VectorIndex, question: str, top_k: int = 5
) -> list[tuple[Document, float]]:
    """Embed with the indexed model and return passages in descending score order."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    from src.embeddings import embed_texts, load_embedding_model

    model = load_embedding_model(index.model_name)
    query = embed_texts([question], model)[0]
    if query.shape != (index.embeddings.shape[1],) or not np.isfinite(query).all():
        raise ValueError("Query embedding is incompatible with the saved index.")
    scores = index.embeddings @ query
    positions = np.argsort(-scores, kind="stable")[:top_k]
    return [(index.chunks[i], float(scores[i])) for i in positions]
