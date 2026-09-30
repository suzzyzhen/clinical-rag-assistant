"""Build a local WHO index or answer questions using its evidence."""

import argparse
import os

from src.config import CHUNKER_RUNS, DEFAULT_CHUNKER_RUN, INDEX_PATH, LLM_MODEL, TOP_K


def build_index(index_path, run):
    from src.data_loading.download_pdfs import download_recent_publications
    from src.data_loading.pdf_loader import load_pdfs_from_folder
    from src.data_loading.web_loader import load_who_fact_sheets
    from src.data_processing.chunking import (
        ChunkerConfig,
        chunk_documents,
        recursive_chunk_documents,
    )
    from src.data_processing.cleaning import clean_documents
    from src.embeddings import embed_documents, load_embedding_model
    from src.vector_store import save_index

    download_recent_publications(wer_count=10, drug_info_count=0, out_dir="data/who")

    pdf_docs = load_pdfs_from_folder(
        folder="data/who",
        source_name="WHO IRIS Publications",
        manifest_path="data/who/manifest_iris.json",
    )
    web_docs = load_who_fact_sheets(
        # num_pages=10,
        num_pages=None,
        manifest_path="data/who/manifest_web.json",
    )

    pdf_docs = [
        doc
        for doc in pdf_docs
        if "weekly epidemiological record" in (doc.metadata.get("title") or "").lower()
    ]
    raw_docs = pdf_docs + web_docs
    cleaned_docs = clean_documents(raw_docs)

    model = load_embedding_model(run["model"])

    def token_count(text: str) -> int:
        return len(
            model.tokenizer(text, add_special_tokens=True, truncation=False)[
                "input_ids"
            ]
        )

    if run["chunker"] == "section_aware":
        cfg = ChunkerConfig(
            target_tokens=run["chunk_size"],
            overlap_tokens=run["chunk_overlap"],
            token_counter=token_count,
        )
        chunks = chunk_documents(cleaned_docs, cfg)
    else:
        chunks = recursive_chunk_documents(
            cleaned_docs,
            model_name=run["model"],
            chunk_size=run["chunk_size"],
            chunk_overlap=run["chunk_overlap"],
            token_counter=token_count,
        )

    chunk_ids, embeddings = embed_documents(chunks, model)
    save_index(index_path, chunks, embeddings, run["model"], run)
    print(f"Saved index to {index_path}")

    print(f"Run: {run['name']}")
    print(f"Raw docs: {len(raw_docs)}")
    print(f"Cleaned docs: {len(cleaned_docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"First chunk id: {chunk_ids[0] if chunk_ids else 'None'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build-index", help="Download WER and fact sheets, then save embeddings"
    )
    build.add_argument("--index-path", default=INDEX_PATH)
    build.add_argument(
        "--run",
        choices=[r["name"] for r in CHUNKER_RUNS],
        default=DEFAULT_CHUNKER_RUN,
    )
    ask = commands.add_parser("ask", help="Answer a question using a saved index")
    ask.add_argument("question")
    ask.add_argument("--index-path", default=INDEX_PATH)
    ask.add_argument("--top-k", type=int, default=TOP_K)
    ask.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args(argv)

    if args.command == "ask":
        if not args.question.strip():
            parser.error("Question must not be empty.")
        if args.top_k < 1:
            parser.error("--top-k must be positive.")
        if not os.environ.get("GEMINI_API_KEY"):
            parser.error("Set GEMINI_API_KEY in your environment or project .env file.")

    from openai import APIError
    from requests import RequestException

    try:
        if args.command == "build-index":
            run = next(r for r in CHUNKER_RUNS if r["name"] == args.run)
            build_index(args.index_path, run)
        else:
            from src.rag_pipeline import answer_question

            result = answer_question(
                args.question, args.index_path, args.top_k, args.model
            )
            print(result["answer"])
            if result["sources"]:
                print("\nSources:")
            for source in result["sources"]:
                page = (
                    f", PDF page {source['page_number']}"
                    if source.get("page_number") is not None
                    else ""
                )
                print(f"[{source['id']}] {source.get('title') or 'Untitled'}{page}")
                print(f"    {source.get('source') or 'Source URL unavailable'}")
    except APIError as exc:
        parser.exit(
            1,
            f"Gemini request failed ({type(exc).__name__}). Check your API key, model access and connection.\n",
        )
    except (OSError, ValueError, RequestException) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
