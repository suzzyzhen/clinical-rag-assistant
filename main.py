from app.config import CHUNKER_RUNS
from app.data_loading.download_pdfs import download_recent_publications
from app.data_loading.pdf_loader import load_pdfs_from_folder
from app.data_loading.web_loader import load_who_fact_sheets
from app.data_processing.chunking import (
    ChunkerConfig,
    chunk_documents,
    recursive_chunk_documents,
)
from app.data_processing.cleaning import clean_documents
from app.embeddings import embed_documents, load_embedding_model


def main() -> None:
    download_recent_publications(wer_count=5, drug_info_count=2, out_dir="data/who")

    pdf_docs = load_pdfs_from_folder(
        folder="data/who",
        source_name="WHO IRIS Publications",
        manifest_path="data/who/manifest_iris.json",
    )
    web_docs = load_who_fact_sheets(
        num_pages=3,
        manifest_path="data/who/manifest_web.json",
    )

    raw_docs = pdf_docs + web_docs
    cleaned_docs = clean_documents(raw_docs)

    run = CHUNKER_RUNS[0]
    model = load_embedding_model(run["model"])

    def token_count(text: str) -> int:
        return len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

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

    print(f"Run: {run['name']}")
    print(f"Raw docs: {len(raw_docs)}")
    print(f"Cleaned docs: {len(cleaned_docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"First chunk id: {chunk_ids[0] if chunk_ids else 'None'}")


if __name__ == "__main__":
    main()
