"""Local embedding utilities for cleaned, chunked WHO source documents."""

from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> SentenceTransformer:
    """Load the embedding model once and pass it to later calls.

    The first invocation downloads the model if it is not already cached.
    """
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Return L2-normalized vectors suitable for cosine-similarity search."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_documents(
    chunks: Sequence[Document],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    """Embed indexable chunk documents and return their stable IDs and vectors."""
    
    chunks_to_embed = [
        chunk for chunk in chunks
        if chunk.metadata.get("is_indexable", True)
    ]
    chunk_ids = []
    for chunk in chunks_to_embed:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Each chunk must have metadata['chunk_id']; run chunk_documents first.")
        chunk_ids.append(str(chunk_id))

    embeddings = embed_texts(
        [chunk.page_content for chunk in chunks_to_embed],
        model,
        batch_size,
    )
    return chunk_ids, embeddings
