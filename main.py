from app.data_loading.download_pdfs import download_recent_publications
from app.data_loading.web_loader import load_who_fact_sheets
from app.data_loading.pdf_loader import load_pdfs_from_folder


# 1. download pdfs from WHO IRIS website for publications

download_recent_publications(wer_count=5, drug_info_count=2, out_dir="data/who")

# 2. load pdfs into Langchain documents
pdf_docs = load_pdfs_from_folder(
    folder="data/who",
    source_name="WHO IRIS Publications",
    manifest_path="data/who/manifest_iris.json",
)

# 3. load WHO fact sheets from the website
web_docs = load_who_fact_sheets(
    num_pages=3, 
    manifest_path="data/who/manifest_web.json")


# 4.cleaning