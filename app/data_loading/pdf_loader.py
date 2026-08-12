import hashlib
from pathlib import Path
from typing import Optional

import fitz
import pdfplumber
from langchain_core.documents import Document
import json

WHO_LICENSE = "CC BY-NC-SA 3.0 IGO"


def _make_document_id(source: str) -> str:
    """Create a stable document ID from the source."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _format_table(table: list[list]) -> str:
    """
    Convert a PDF table into text.

    No semantic cleaning is performed here.
    """
    rows = []
    for row in table:
        cleaned_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(" | ".join(cleaned_row))

    if not rows:
        return ""

    return "[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"


def _extract_page_content(fitz_page, plumber_page) -> str:
    """
    Extract raw text and tables from one PDF page.

    PyMuPDF is used for normal text.
    pdfplumber is used for table detection/extraction.
    Cleaning is intentionally deferred to cleaning.py.
    """
    tables = plumber_page.find_tables()
    table_bboxes = [table.bbox for table in tables]
    table_texts = [
        text for text in
        (_format_table(t) for t in plumber_page.extract_tables())
        if text.strip()
    ]

    # --- Extract prose ---
    if not table_bboxes:
        prose = fitz_page.get_text("text")
    else:
        blocks = fitz_page.get_text("blocks")
        prose = "\n".join(
            block[4] for block in blocks
            if not any(
                fitz.Rect(block[:4]).intersects(fitz.Rect(bbox))
                for bbox in table_bboxes
            )
        )

    # --- Combine prose + tables ---
    sections = []
    if prose.strip():
        sections.append(prose)
    sections.extend(table_texts)

    return "\n\n".join(sections)


def load_pdf(
    pdf_path: str,
    source_name: str,
    title: Optional[str] = None,
    url: Optional[str] = None,
    license: str = WHO_LICENSE,
    manifest_entry: Optional[dict] = None, 
) -> list[Document]:
    pdf_path_obj = Path(pdf_path)
    # source = manifest_entry.get("source") or url or str(pdf_path_obj)
    document_id = _make_document_id(str(pdf_path_obj))
    docs = []

    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        # pdf_metadata = fitz_doc.metadata or {}
        n_pages = len(fitz_doc)

        for page_num, (fitz_page, plumber_page) in enumerate(
            zip(fitz_doc, plumber_doc.pages), start=1
        ):
            page_content = _extract_page_content(fitz_page, plumber_page)
            if not page_content.strip():
                continue

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "document_id":    document_id,
                    "source":         manifest_entry.get("item_url"),
                    "source_type":    "pdf",
                    "source_name":    source_name,
                    "title":          manifest_entry.get("title"),
                    "description":    manifest_entry.get("description"),
                    "language":       manifest_entry.get("language"),
                    "published_date": manifest_entry.get("published_date"),
                    "license":        manifest_entry.get("license") or license,
                    "page_number":    page_num,
                    "n_pages":        n_pages,
                    # "pdf_url":        manifest_entry.get("pdf_url"),
                },
            ))

    return docs


def load_pdfs_from_folder(
    folder: str,
    source_name: str,
    license: str = WHO_LICENSE,
    manifest_path: Optional[str] = None,
) -> list[Document]:
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    # Load manifest once and index by local_path filename
    manifest_index = {}
    if manifest_path and Path(manifest_path).exists():
        entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_index = {
            Path(entry["local_path"]).name: entry
            for entry in entries
            if entry.get("local_path")
        }
        print(f"Loaded {len(manifest_index)} manifest entries from {manifest_path}")

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            manifest_entry = manifest_index.get(pdf_path.name, {})
            pdf_docs = load_pdf(
                pdf_path=str(pdf_path),
                source_name=source_name,
                license=license,
                manifest_entry=manifest_entry,
            )
            docs.extend(pdf_docs)
            print(f"  Loaded {len(pdf_docs)} pages")
        except Exception as e:
            print(f"  ! Failed to load {pdf_path.name}: {e}")

    print(f"Loaded {len(docs)} page documents from {len(pdf_files)} PDFs.")
    return docs