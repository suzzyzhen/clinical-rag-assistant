Project overview

# World Health RAG Assistant

A retrieval-augmented generation (RAG) system for answering clinical questions using authoritative healthcare guidelines from the World Health Organization (WHO).

The project demonstrates an end-to-end healthcare AI pipeline: PDF ingestion, HTML parsing, document processing, semantic search, and grounded answer generation with citations.

---

## Features

* **Guideline-based question answering** using WHO documents
* **Page-level document ingestion** for precise citations
* **Table-aware PDF extraction** using PyMuPDF and pdfplumber
* **Semantic retrieval** with vector embeddings
* **Grounded responses** that reference the source guideline and page number
* Modular pipeline designed for experimentation with different embedding models and LLMs

---

## Example

**Question**

> According to the latest WHO Weekly Epidemiological Record, what are the rising public health concerns, what are the suggested treatments accroding to the WHO fact sheets?

**Answer**

The assistant retrieves the relevant WHO Weekly Epidemiological Record and generates a response based only on the retrieved evidence, including the page reference used to produce the answer.

---

## Project Structure

```text
clinical-rag-assistant/
├── data/
│   ├── raw/                 # Original guideline PDFs
│   ├── processed/           # Cleaned / intermediate data
│   └── vectorstore/         # Vector database files
│
├── src/
│   ├── loader.py            # PDF ingestion (1 Document per page)
│   ├── parser.py            # Cleaning and chunking logic
│   ├── embeddings.py        # Embedding generation
│   ├── retriever.py         # Vector search
│   ├── rag_pipeline.py      # End-to-end RAG workflow
│   └── schema.py            # Shared enums / metadata types
│
├── notebooks/               # Experiments and evaluation
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

---

## Pipeline

```text
Epidemiological Records (PDF) | WHO Fact Sheets | WHO Drug Information
      │
      ▼
PDF Loader (PyMuPDF + pdfplumber) | LangChain's WebBaseLoader 
      │
      ▼
LangChain Documents (1 per page)
      │
      ▼
Text Cleaning
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Database
      │
      ▼
Retriever
      │
      ▼
LLM
      │
      ▼
Grounded Answer + Citation
```

---

## Data Sources
The Clinical RAG Assistant retrieves information from multiple authoritative World Health Organization (WHO) resources:
1. WHO Fact Sheets 
    Format: HTML
    Public health summaries covering diseases, conditions, risk factors, and global health topics. These are ingested directly from the WHO website using LangChain's `WebBaseLoader`.
2. WHO Drug Information 2026 Issues 
    Format: PDF
    Quarterly publications containing guidance on medicine regulation, pharmacovigilance, bioequivalence, pharmaceutical quality, and regulatory science.
3. WHO Weekly Epidemiological Record (WER) 
    Format: PDF 


## Document Ingestion

Each page is converted into **one LangChain `Document` per page**.

Example metadata:

```python
{
    "title": "Preterm labour and birth",
    "page": 18,
    "source_name": "WHO",
    "doc_type": "LITERATURE",
    "published_date": "26 June 2026"
}
```

This allows every retrieved chunk to retain its original page reference.

---

## Installation

### Prerequisites

* Python 3.11+
* `uv` package manager

Install `uv` if you don't already have it:

```bash
pip install uv
```

or follow the official installation instructions:

https://docs.astral.sh/uv/

### Clone the repository

```bash
git clone https://github.com/<your-username>/clinical-rag-assistant.git

cd clinical-rag-assistant
```

### Create a virtual environment

```bash
uv venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
uv sync
```


## Usage

Place guideline PDFs in:

```text
data/who/
```

Example:

```text
data/who/
├── WER-101-25-eng.pdf
├── WHO_drug_2026_1.pdf
```

Load documents:

```python
from src.loader import load_pdfs_from_folder
from src.schema import DocType

documents = load_pdfs_from_folder(
    folder="data/raw",
    source_name="NICE",
    doc_type=DocType.GUIDELINE
)
```

The loader returns a list of LangChain `Document` objects, one for each page.

---

## Current Tech Stack

* **Python**
* **LangChain**
* **PyMuPDF (fitz)**
* **pdfplumber**
* **LLM APIs**
* **Vector database **

---

## Roadmap

* [ ] Semantic chunking using document headings
* [ ] Better table extraction and normalization
* [ ] Figure and flowchart transcription
* [ ] Hybrid retrieval (BM25 + vector search)
* [ ] LLM-as-a-Judge evaluation pipeline
* [ ] FastAPI deployment
* [ ] Docker support
* [ ] CI/CD with GitHub Actions

---

## Why This Project Exists

Large language models can answer many medical questions, but clinical practice requires **traceable evidence**.

This project focuses on:

* grounding answers in authoritative guidelines,
* preserving document provenance,
* enabling page-level citations,
* and building a workflow suitable for healthcare AI applications where transparency and auditability matter.

---

## Disclaimer

This project is for research and educational purposes only. It is **not a medical device** and should not be used as a substitute for professional clinical judgment or official guideline consultation.



data things to keep in mind:
tables/charts on HTML pages are not read 
pdf charts does not have meaning/ if want to extract info need to use vLLM (describe_image_with_llm)
