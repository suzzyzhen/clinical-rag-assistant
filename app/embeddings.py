"""Local embedding utilities for cleaned, chunked WHO source documents."""

from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str | None = None,
) -> SentenceTransformer:
    """Load an embedding model."""
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Return L2-normalized vectors suitable for cosine-similarity search."""
    if isinstance(texts, str):
        raise TypeError("embed_texts expects a sequence of strings, not a single string. Wrap the text in a list.")
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
    """Embed chunk documents and return their stable IDs and vectors."""
    chunk_ids = []
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Each chunk must have metadata['chunk_id']; run chunk_documents first.")
        chunk_ids.append(str(chunk_id))

    embeddings = embed_texts(
        [chunk.page_content for chunk in chunks],
        model,
        batch_size,
    )
    return chunk_ids, embeddings
