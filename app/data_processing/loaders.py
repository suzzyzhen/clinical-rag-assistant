"""
loaders.py
Turns source files into RawDocument objects.

Two loaders are included:
- load_pdf(): for guideline/policy PDFs (uses PyMuPDF)
- load_pmc_xml(): for PMC OA Subset JATS XML (much cleaner than PDF --
  see the note in the README about preferring this source where possible)
"""
from pathlib import Path
from typing import Optional

import fitz
import pdfplumber

from langchain_core.documents import Document


def _format_table(table: list[list]) -> str:
    return "\n".join(
        " | ".join(str(cell or "").strip().replace("\n", " ") for cell in row)
        for row in table
    )


def _extract_page_content(pdf_path: str) -> str:
    page_texts = []
    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        for fitz_page, plumber_page in zip(fitz_doc, plumber_doc.pages):
            table_bboxes = [t.bbox for t in plumber_page.find_tables()]
            table_texts = [_format_table(t) for t in plumber_page.extract_tables()]

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

            page_texts.append("\n\n".join(filter(None, [prose] + table_texts)))

    return "\n\n".join(page_texts)


def load_pdf(
    pdf_path: str,
    source_name: str,
    title: Optional[str] = None,
    url: Optional[str] = None,
    license: str = "unknown",
) -> list[Document]:
    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        resolved_title = title or fitz_doc.metadata.get("title") or Path(pdf_path).stem
        pdf_metadata = fitz_doc.metadata
        n_pages = len(fitz_doc)
        docs = []

        for page_num, (fitz_page, plumber_page) in enumerate(zip(fitz_doc, plumber_doc.pages), start=1):
            table_bboxes = [t.bbox for t in plumber_page.find_tables()]
            table_texts = [_format_table(t) for t in plumber_page.extract_tables()]

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

            page_content = "\n\n".join(filter(None, [prose] + table_texts))
            if not page_content.strip():
                continue  # skip blank pages (common for scanned cover pages)

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "source":         url or str(pdf_path),
                    "title":          resolved_title,
                    "description":    pdf_metadata.get("subject"),
                    "language":       pdf_metadata.get("language"),
                    "source_name":    source_name,
                    "license":        license,
                    "published_date": pdf_metadata.get("creationDate"),
                    "n_pages":        n_pages,
                    "page_number":    page_num,
                },
            ))

    return docs


def load_pdfs_from_folder(
    folder: str,
    source_name: str,
    license: str = "unknown",
) -> list[Document]:
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            docs.extend(load_pdf(
                pdf_path=str(pdf_path),
                source_name=source_name,
                license=license,
            ))
        except Exception as e:
            print(f"  ! failed to load {pdf_path.name}: {e}")

    print(f"Loaded {len(docs)} of {len(pdf_files)} PDFs from {folder}")
    return docs