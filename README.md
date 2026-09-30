# World Health RAG Assistant

A retrieval-augmented generation (RAG) application for exploring health questions using WHO publications and fact sheets. It retrieves passages from a local document index and asks Google Gemini to generate an answer with numbered source citations.

This project explores how source provenance, document extraction, chunking, and embedding choices affect evidence-based question answering. It provides both a command-line interface and a local Gradio interface.

For research and educational use only. This is not a medical device or clinical decision-support system, and generated answers require checking against the original sources.

## How it works

```text
WHO Weekly Epidemiological Record PDFs + WHO fact-sheet HTML
                         ↓
      Text/table extraction and source metadata
                         ↓
           Cleaning and token-based chunking
                         ↓
          Local sentence-transformer embeddings
                         ↓
              NumPy/JSON vector index
                         ↓
Question → query embedding → top-k cosine retrieval
                         ↓
       Gemini answer with numbered source citations
```

PDF ingestion creates one document per nonempty page using PyMuPDF for text and pdfplumber for tables. Web ingestion creates one document per fact sheet using requests and BeautifulSoup. Cleaning removes common extraction artifacts and uses heuristics to trim web navigation and repeated PDF boilerplate.

Chunks retain source titles, URLs, publication dates, and 1-based PDF page numbers where available. PDF page numbers refer to the file's page position, which may differ from printed publication numbering.

At query time, the application retrieves five passages by default, removes exact duplicate passage text, and supplies up to 24,000 characters of evidence to the generation model. The prompt asks the model to use only that evidence and report when it is insufficient. The application checks citation IDs and returns the cited sources alongside the answer.

## Quick start

### 1. Install dependencies

Requires Python 3.11+ and `uv`. Run commands from the project root.

```bash
git clone https://github.com/suzzyzhen/clinical-rag-assistant.git
cd clinical-rag-assistant
uv sync
```

The commands below use `uv run`, so you do not need to activate the virtual environment manually.

### 2. Build the local index

```bash
uv run python main.py build-index
```

The default build:

- Requests ten recent WHO Weekly Epidemiological Record (WER) issues and downloads their PDFs into `data/who/`.
- Loads local PDFs and retains those whose manifest titles identify them as WER publications.
- Discovers fact-sheet links from the WHO fact-sheet index page and fetches all discovered links, without a page-count limit.
- Cleans, chunks, and embeds the documents, then writes `embeddings.npy` and `chunks.json` into `data/vectorstore/`.

The first build requires internet access for WHO content and the embedding model download. No Gemini API key is needed for indexing. Subsequent questions reuse the saved index; rebuilding replaces it. Avoid building and querying concurrently against the same index directory.

### 3. Configure answer generation

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-api-key
LLM_MODEL=gemini-2.5-flash
```

`.env` is gitignored. Generation uses the `openai` Python client configured to send requests to Google's Gemini endpoint. Your question and retrieved excerpts are sent to Google; embedding and vector search run locally. Do not enter private patient information into this demo.

### 4. Ask a question

```bash
uv run python main.py ask "What causes anaemia?"
uv run python main.py ask "What causes anaemia?" --top-k 3
```

These are example queries, not recorded evaluation results. Answers depend on the content present in your index.

For the browser interface:

```bash
uv run python gradio_app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`. The interface uses the same saved index and answer pipeline as the CLI.

You can also call the pipeline from Python:

```python
from src.rag_pipeline import answer_question

result = answer_question("What causes anaemia?")
print(result["answer"])
print(result["sources"])
```

## Configuration and experiments

The default build uses `pubmedbert_section_450`. Available runs are defined in [src/config.py](src/config.py):

| Run | Embedding model | Chunker | Target tokens | Overlap tokens |
| --- | --- | --- | --- | --- |
| `minilm_section_220` | `sentence-transformers/all-MiniLM-L6-v2` | Section-aware | 220 | 30 |
| `minilm_recursive_220` | `sentence-transformers/all-MiniLM-L6-v2` | Recursive | 220 | 30 |
| `pubmedbert_section_450` | `NeuML/pubmedbert-base-embeddings` | Section-aware | 450 | 60 |
| `pubmedbert_recursive_450` | `NeuML/pubmedbert-base-embeddings` | Recursive | 450 | 60 |

Section-aware sizes are targets, not strict limits: long sentences can exceed the target. Token counts use the selected model's tokenizer.

Use a separate index directory to compare configurations:

```bash
uv run python main.py build-index --run pubmedbert_recursive_450 --index-path data/pubmedbert-index
uv run python main.py ask "What causes anaemia?" --index-path data/pubmedbert-index
```

Add custom index directories to `.gitignore`. The saved index records its embedding model and chunking configuration; queries use that recorded model. Changing embeddings or chunking requires rebuilding the index. Changing the generation model does not.

| Setting | Purpose |
| --- | --- |
| `INDEX_PATH` | Default index directory; defaults to `data/vectorstore` |
| `LLM_MODEL` | Generation model; defaults to `gemini-2.5-flash` |
| `ask --model` | Override the generation model for one CLI query |
| `ask --top-k` | Number of retrieved passages; defaults to 5 |
| `build-index --run` | Select an embedding and chunking configuration |

## Design decisions and tradeoffs

| Choice | Benefit | Tradeoff |
| --- | --- | --- |
| WHO publications and fact sheets | A focused corpus with identifiable sources | Limited topic coverage; publication dates and source authority do not guarantee that a retrieved passage answers the question |
| One document per PDF page | Preserves page references for inspection | Context and tables spanning pages can be split |
| Section-aware and recursive chunking | Supports comparing structural boundaries with token-based splitting | Heading detection is heuristic; overlap adds duplicate context and storage |
| Local MiniLM and PubMedBERT embeddings | Enables comparison of general-purpose and biomedical representations without an embedding API | Models differ in compute requirements and tokenization; biomedical specialization alone does not establish better retrieval |
| NumPy/JSON index with exact cosine search | Simple persistence and inspectable retrieval without a database service | Loads the index into memory and scores every vector; intended for a small corpus |
| Dense top-k retrieval | A straightforward semantic-search baseline | No keyword retrieval, reranking, or calibrated relevance threshold; similar passages may still be irrelevant |
| Gemini generation with citation checks | Produces readable answers linked to retrieved evidence | Requires a remote API; valid citation IDs do not prove that the cited passages support every claim |

## Evaluation

The [chunking and embedding analysis notebook](notebooks/chunking_and_embedding_analysis.ipynb) compares chunk/token distributions and retrieval across the configured runs. It includes a simple hit@k check using expected terms; this is a limited retrieval signal, not a measure of clinical correctness or citation support.

Two evaluation datasets provide questions, reference answers, expected terms, and source metadata:

- [WER evaluation questions](notebooks/eval_questions_wer.json): 30 questions.
- [Web evaluation questions](notebooks/eval_questions_web.json): 20 questions.

For a manual end-to-end check, sample questions from both datasets and add questions outside the corpus. Inspect whether retrieval finds the expected evidence, whether the answer preserves numerical details and dates, and whether each citation supports the associated claim. Check that unsupported questions receive an insufficient-evidence response.

The datasets refer to particular sources. A fresh download can change the corpus, so retain the source snapshot and record the run configuration when comparing results. No clinical validation or measured superiority of the default configuration is claimed here.

Tests are currently excluded from version control. If you have the local `tests/` directory, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q
```

Disabling plugin autoload avoids unrelated installed pytest plugins. A fresh clone may not include these local tests.

## Data sources and provenance

The default build uses WER PDFs and WHO fact-sheet HTML. The downloader also supports WHO Drug Information publications, but the default CLI build requests zero of those and filters PDF inputs to WER.

[src/data_loading/web_loader.py](src/data_loading/web_loader.py) also exposes `load_who_fact_sheets_api` as an alternative English-content loader. The default build calls the separate HTML loader, `load_who_fact_sheets`. HTML extraction and API content can differ; the HTML path flattens table text, while the API path includes explicit table-row formatting.

Source manifests in `data/who/manifest_iris.json` and `data/who/manifest_web.json` record source URLs, titles, dates, local PDF paths where applicable, and licence labels. Runtime document metadata can retain additional licence-detection evidence. A null licence means that no licence label was established by ingestion; source ownership alone does not establish reuse rights. Existing PDF manifest labels can be reused without rechecking the original notice.

Downloaded PDFs and the default vector index are gitignored. Updating a manifest does not update an existing index; rebuild it to incorporate refreshed content or metadata.

## Known limitations

- **Snapshot coverage:** the index is not a complete or continuously updated WHO collection. It cannot reliably answer requests for the “latest” guidance without evidence establishing recency.
- **Extraction quality:** complex PDF layouts, tables, and cleaning heuristics can lose or reorder information. There is no OCR or visual interpretation of charts, images, or flowcharts.
- **Chunk boundaries:** PDF page boundaries and heuristic section detection can separate evidence from its context. Oversized chunks may be truncated by the embedding model.
- **Answer reliability:** the system validates citation IDs, not factual support. Insufficient-evidence handling depends on the generation model when retrieval returns passages.
- **Operational scope:** this is a local research application with external ingestion and generation dependencies, not a production clinical service.

## Project structure

```text
clinical-rag-assistant/
├── main.py                   # Build-index and ask CLI
├── gradio_app.py             # Local browser interface
├── src/
│   ├── config.py             # Defaults and experiment configurations
│   ├── data_loading/         # PDF/HTML/API ingestion and licence metadata
│   ├── data_processing/      # Cleaning and chunking
│   ├── embeddings.py         # Local sentence-transformer embeddings
│   ├── vector_store.py       # NumPy/JSON persistence and cosine retrieval
│   └── rag_pipeline.py       # Generation and citation-ID checks
├── data/who/                 # Source manifests and local PDFs
├── data/vectorstore/         # Generated index (gitignored)
├── notebooks/                # Analysis and evaluation datasets
└── pyproject.toml            # Dependencies and development tools
```

## Next steps

- Compare retrieval configurations using source/page matches as well as expected terms.
- Evaluate answer faithfulness and citation support separately from retrieval quality.
- Explore hybrid keyword/vector retrieval and reranking.
- Improve table handling, heading detection, and extraction of visual evidence.
- Establish reproducible corpus snapshots and restore tracked tests before adding CI.
