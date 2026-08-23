import re
from collections import Counter
from itertools import groupby
from typing import Optional

from langchain_core.documents import Document


_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(of\s*\d{1,4})?\s*$", re.IGNORECASE)
_MULTI_WHITESPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WEB_START_MARKERS = ("Key facts", "Overview")
_WEB_END_MARKERS = ("Related", "Related topics")
_NAV_ANCHOR_MIN_CHARS = 120
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def trim_navigation(text: str, min_anchor_chars: int = _NAV_ANCHOR_MIN_CHARS) -> str:
    lines = text.split("\n")

    for i, line in enumerate(lines):
        if line.strip() in _WEB_START_MARKERS:
            return "\n".join(lines[i:])

    # Fallback: no known marker found, use length heuristic
    for i, line in enumerate(lines):
        if len(line.strip()) >= min_anchor_chars:
            return "\n".join(lines[i:])

    return text


def trim_web_footer(text: str) -> str:
    """Trim trailing footer content from a web page."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in _WEB_END_MARKERS:
            return "\n".join(lines[:i])
    return text


def _drop_repeated_boilerplate(lines: list[str], boilerplate: set[str]) -> list[str]:
    """Remove lines that match known boilerplate."""
    return [line for line in lines if line.strip() not in boilerplate]


def _find_repeated_boilerplate(documents: list[Document], threshold: int) -> set[str]:
    """Find lines repeated more than `threshold` times in one document."""
    counts = Counter()
    for doc in documents:
        for line in doc.page_content.split("\n"):
            stripped = line.strip()
            if stripped:
                counts[stripped] += 1

    return {line for line, count in counts.items() if count > threshold}


def _find_repeated_boilerplate_by_document(
    documents: list[Document],
    threshold: int,
) -> dict[str, set[str]]:
    """Find repeated boilerplate separately for each document_id."""
    boilerplate_by_doc: dict[str, set[str]] = {}

    sorted_docs = sorted(documents, key=lambda d: d.metadata.get("document_id", ""))
    for doc_id, group in groupby(sorted_docs, key=lambda d: d.metadata.get("document_id", "")):
        boilerplate_by_doc[doc_id] = _find_repeated_boilerplate(list(group), threshold)

    return boilerplate_by_doc


def clean_text(
    raw: str,
    is_web: bool = False,
    boilerplate: Optional[set[str]] = None,
) -> str:
    """Clean extracted text and normalize whitespace."""
    text = raw

    # --- Web-specific cleaning ---
    if is_web:
        text = trim_navigation(text)
        text = trim_web_footer(text)

    # --- Control characters ---
    text = _CONTROL_CHARS.sub("", text)

    # --- Rejoin words broken by hyphen e.g. "hyper-\ntension" -> "hypertension" ---
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # --- Remove page-number lines ---
    lines = text.split("\n")
    lines = [line for line in lines if not _PAGE_NUMBER_LINE.match(line)]

    # --- Remove repeated headers/footers (e.g. running title on every page) ---
    if boilerplate:
        lines = _drop_repeated_boilerplate(lines, boilerplate)

    # --- Strip trailing whitespace ---
    text = "\n".join(lines)
    text = _TRAILING_WHITESPACE.sub("", text)

    # --- Normalize whitespace ---
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def clean_documents(
    documents: list[Document],
    drop_repeated_lines_threshold: int = 3,
) -> list[Document]:
    """Clean documents and drop any that become empty."""
    if not documents:
        return []

    # --- Identify PDF boilerplate, scoped per document ---
    pdf_documents = [doc for doc in documents if doc.metadata.get("source_type") == "pdf"]
    boilerplate_by_doc = (
        _find_repeated_boilerplate_by_document(pdf_documents, threshold=drop_repeated_lines_threshold)
        if pdf_documents
        else {}
    )

    # --- Clean each document ---
    cleaned_documents = []
    for doc in documents:
        is_web = doc.metadata.get("source_type") == "web"
        doc_id = doc.metadata.get("document_id", "")

        cleaned_text = clean_text(
            doc.page_content,
            is_web=is_web,
            boilerplate=None if is_web else boilerplate_by_doc.get(doc_id, set()),
        )

        if not cleaned_text:
            continue

        cleaned_documents.append(Document(page_content=cleaned_text, metadata=doc.metadata))

    return cleaned_documents
