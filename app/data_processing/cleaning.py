import re
from collections import Counter
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
    """Removes the trailing footer block from web content by finding the
    first line matching a known content-end marker (e.g. "Related") and
    discarding everything from that point onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in _WEB_END_MARKERS:
            return "\n".join(lines[:i])
    return text


def clean_text(
    raw: str,
    is_web: bool = False,
) -> str:
    """
    Cleans extracted text through the following steps:
      Remove web navigation/footer blocks (if is_web=True)
      Remove control characters
      Rejoin hyphenated words split across a line break
      Remove page-number-only lines
      Strip trailing whitespace
      Normalize whitespace
    """
    text = raw

    # --- Web-specific cleaning ---
    if is_web:
        text = trim_navigation(text)
        text = trim_web_footer(text)

    # --- Control characters ---
    text = _CONTROL_CHARS.sub("", text)

    # --- Rejoin words broken by hyphen e.g. "hyper-\ntension" -> "hypertension" ---
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # # --- Remove page-number lines ---
    lines = text.split("\n")
    lines = [line for line in lines if not _PAGE_NUMBER_LINE.match(line)]

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
    """
    Cleans a list of LangChain Documents, handling web and PDF sources
    differently:
      - Web documents: cleaned individually (nav/footer trimming per page).
    """
    if not documents:
        return []

    # --- Clean each document ---
    cleaned_documents = []
    for doc in documents:
        is_web = doc.metadata.get("source_type") == "web"

        cleaned_text = clean_text(
            doc.page_content,
            is_web=is_web,
        )

        if not cleaned_text:
            continue

        cleaned_documents.append(Document(page_content=cleaned_text, metadata=doc.metadata))

    return cleaned_documents


