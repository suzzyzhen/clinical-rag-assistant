import hashlib
from pathlib import Path
from typing import Optional

import fitz
import pdfplumber
from langchain_core.documents import Document


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
) -> list[Document]:
    """
    Load a PDF as page-level LangChain Documents.

    Each non-empty page becomes one Document.
    No text cleaning or chunking is performed here.
    """
    pdf_path_obj = Path(pdf_path)
    source = url or str(pdf_path_obj)
    document_id = _make_document_id(source)
    docs = []

    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        pdf_metadata = fitz_doc.metadata or {}
        resolved_title = title or pdf_metadata.get("title") or pdf_path_obj.stem
        n_pages = len(fitz_doc)

        # --- Page-level extraction ---
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
                    "source":         source,
                    "source_type":    "pdf",
                    "source_name":    source_name,
                    "title":          resolved_title,
                    "description":    pdf_metadata.get("subject"),
                    "language":       pdf_metadata.get("language"),
                    "published_date": None,
                    "license":        license,
                    "page_number":    page_num,
                    "n_pages":        n_pages,
                },
            ))

    return docs


def load_pdfs_from_folder(
    folder: str,
    source_name: str,
    license: str = WHO_LICENSE,
) -> list[Document]:
    """
    Load all PDFs from a folder.

    Returns page-level LangChain Documents.
    """
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            pdf_docs = load_pdf(
                pdf_path=str(pdf_path),
                source_name=source_name,
                license=license,
            )
            docs.extend(pdf_docs)
            print(f"  Loaded {len(pdf_docs)} pages")
        except Exception as e:
            print(f"  ! Failed to load {pdf_path.name}: {e}")

    print(f"Loaded {len(docs)} page documents from {len(pdf_files)} PDFs.")
    return docs