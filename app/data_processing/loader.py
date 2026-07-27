from pathlib import Path
from pypdf import PdfReader
from dataclasses import dataclass

@dataclass
class RawDocument:
    filename: str
    page_number: int
    text: str

def load_pdf(file_path: Path):
    reader = PdfReader(file_path)

    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        documents.append(
            RawDocument(
                filename=file_path.name,
                page_number=page_num + 1,
                text=text,
            )
        )

    return documents

def load_directory(directory: Path):
    all_documents = []

    for pdf in directory.glob("*.pdf"):
        all_documents.extend(load_pdf(pdf))

    return all_documents
