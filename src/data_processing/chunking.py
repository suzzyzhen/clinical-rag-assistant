import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.documents import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_HEADER_LINE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z0-9 /&\-]{4,60})$")


def _default_token_counter(text: str) -> int:
    """Fallback heuristic when no tokenizer is provided."""
    return max(1, int(len(text) / 4.0))


@dataclass
class ChunkerConfig:
    target_tokens: int = 350
    overlap_tokens: int = 60
    token_counter: Callable[[str], int] = field(default=_default_token_counter)


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into section-title and section-body pairs."""
    lines = text.split("\n")
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in lines:
        if _HEADER_LINE.match(line.strip()) and len(line.strip()) < 80:
            if current_body:
                sections.append((current_title, current_body))
            current_title = line.strip().lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, current_body))

    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def _sentence_chunks(paragraph: str, cfg: ChunkerConfig) -> list[str]:
    """Split a paragraph into overlapping sentence-based chunks."""
    sentences = _SENTENCE_SPLIT.split(paragraph)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_tokens = cfg.token_counter(sent)
        if current_len + sent_tokens > cfg.target_tokens and current:
            chunks.append(" ".join(current))
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                s_tokens = cfg.token_counter(s)
                if overlap_len + s_tokens > cfg.overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_len += s_tokens
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(text: str, cfg: ChunkerConfig | None = None) -> list[dict]:
    """Chunk text into section-aware chunk dictionaries."""
    cfg = cfg or ChunkerConfig()
    results: list[dict] = []

    for section_title, body in _split_into_sections(text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        merged: list[str] = []
        buf = ""
        for p in paragraphs:
            if cfg.token_counter(buf + " " + p) < cfg.target_tokens:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        for para in merged:
            for chunk_text in _sentence_chunks(para, cfg):
                results.append({
                    "text": chunk_text.strip(),
                    "section_title": section_title,
                    "n_tokens_approx": cfg.token_counter(chunk_text),
                })

    return results


def _build_chunk_document(
    source_document: Document,
    chunk_text: str,
    chunk_index: int,
    token_counter: Callable[[str], int],
    section_title: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Build a chunk document with consistent metadata."""
    document_id = str(
        source_document.metadata.get("document_id")
        or source_document.metadata.get("source")
        or "document"
    )
    page_marker = source_document.metadata.get("page_number")
    page_marker = "web" if page_marker is None else str(page_marker)
    chunk_metadata = dict(source_document.metadata if metadata is None else metadata)
    chunk_metadata.update({
        "chunk_id": f"{document_id}:{page_marker}:{chunk_index}",
        "chunk_index": chunk_index,
        "section_title": section_title,
        "n_tokens_approx": token_counter(chunk_text),
    })
    return Document(page_content=chunk_text, metadata=chunk_metadata)


def chunk_documents(
    documents: list[Document],
    cfg: ChunkerConfig | None = None,
) -> list[Document]:
    """Chunk LangChain documents while preserving metadata."""
    cfg = cfg or ChunkerConfig()
    chunks: list[Document] = []

    for document in documents:
        for chunk_index, chunk in enumerate(chunk_document(document.page_content, cfg)):
            chunks.append(_build_chunk_document(
                source_document=document,
                chunk_text=chunk["text"],
                chunk_index=chunk_index,
                token_counter=cfg.token_counter,
                section_title=chunk["section_title"],
            ))

    return chunks


def make_token_counter(model_name: str) -> Callable[[str], int]:
    """Build a token counter from a sentence-transformers tokenizer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    def count(text: str) -> int:
        encoded = model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    return count


def make_chunker_config(
    model_name: str,
    target_tokens: int = 350,
    overlap_tokens: int = 60,
) -> ChunkerConfig:
    """Build a chunker config backed by a model tokenizer."""
    return ChunkerConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        token_counter=make_token_counter(model_name),
    )


def make_recursive_splitter(
    model_name: str,
    chunk_size: int = 350,
    chunk_overlap: int = 60,
    safety_margin: float = 0.85,
):
    """Build a recursive splitter sized with a model tokenizer."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_limit = tokenizer.model_max_length

    if chunk_size is None:
        chunk_size = int(model_limit * safety_margin)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def recursive_chunk_documents(
    documents: list[Document],
    model_name: str,
    chunk_size: int = 350,
    chunk_overlap: int = 60,
    token_counter: Callable[[str], int] | None = None,
) -> list[Document]:
    """Chunk LangChain documents with the recursive splitter."""
    splitter = make_recursive_splitter(
        model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    token_counter = token_counter or make_token_counter(model_name)
    chunks: list[Document] = []

    for document in documents:
        for chunk_index, chunk in enumerate(splitter.split_documents([document])):
            chunks.append(_build_chunk_document(
                source_document=document,
                chunk_text=chunk.page_content,
                chunk_index=chunk_index,
                token_counter=token_counter,
                section_title=None,
                metadata=chunk.metadata,
            ))

    return chunks
