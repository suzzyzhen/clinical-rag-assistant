import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import random

WHO_LICENSE = "CC BY-NC-SA 3.0 IGO"
DEFAULT_HEADERS = {
    "User-Agent": "clinical-rag-portfolio-project/1.0 (personal, non-commercial use)"
}
REQUEST_DELAY_SECONDS = 1.0

def _find_meta(url: str, headers: dict) -> dict:
    """Fetches a single WHO fact sheet page and pulls the fields we need
    for the manifest. Returns a dict with the same keys as the manifest
    schema, so the caller can merge it with what LangChain provides.

 
    Date resolution order:
      1. <meta name="citation_date"> or <meta name="dc.date">
      2. First element with class containing "date" whose text looks like a date
      3. Regex scan of the full body text for "DD Month YYYY"
    """
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! could not fetch {url} for metadata: {e}")
        return {"title": None, "published_date": None, "pdf_url": None}
 
    soup = BeautifulSoup(resp.text, "html.parser")

    published_date = None
    for meta_name in ("citation_date", "dc.date", "date"):
        meta = soup.find("meta", attrs={"name": meta_name})
        if meta and meta.get("content"):
            published_date = meta["content"].strip()
            break
    if not published_date:
        date_el = soup.find(class_=re.compile(r"date", re.I))
        if date_el:
            text = date_el.get_text(strip=True)
            if re.search(r"\b\d{1,2} \w+ \d{4}\b", text):
                published_date = text
    if not published_date:
        body_text = soup.get_text()
        match = re.search(r"\b\d{1,2} \w+ \d{4}\b", body_text)
        if match:
            published_date = match.group(0)
 
    return {"published_date": published_date}


def load_who_fact_sheets(
    source_url: str = "https://www.who.int/news-room/fact-sheets",
    target_prefix: str = "https://www.who.int/news-room/fact-sheets/detail/",
    headers: dict = None,
    manifest_path: str = "../../data/who/manifest.json",
    num_pages: int = None,
) -> tuple[list, list[dict]]:
    """Scrapes the WHO fact-sheets index for links to individual fact
    sheets, loads each one via LangChain, and builds a manifest.json.

    Returns (docs, manifest) where:
      - docs    : list of LangChain Document objects ready for chunking/embedding
      - manifest: list of dicts written to manifest_path (same schema as the
                  publications downloader so everything stays consistent)
    """
    headers = headers or DEFAULT_HEADERS

    # ------------------------------------------------------------------ #
    # Find fact-sheet URLs from the index page                           #
    # ------------------------------------------------------------------ #
    print(f"Fetching index page: {source_url}")
    response = requests.get(source_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    filtered_links = set()
    for anchor in soup.find_all("a", href=True):
        full_url = urljoin(source_url, anchor["href"])
        if full_url.startswith(target_prefix):
            filtered_links.add(full_url)

    urls_to_load = list(filtered_links)
    total_found = len(urls_to_load)
    print(f"Number of fact sheets url's found: {len(urls_to_load)}.")
    
    if num_pages is not None and num_pages > 0:
        sample_size = min(num_pages, total_found)
        urls_to_load = random.sample(urls_to_load, sample_size)
    print(f"Filtered down to {len(urls_to_load)} target fact sheets.")

    if not urls_to_load:
        print("No matching fact sheet links found.")
        return [], []

    # ------------------------------------------------------------------ #
    #          load full page content via LangChain                      #
    # ------------------------------------------------------------------ #
    print("Loading fact sheets via LangChain...")
    loader = WebBaseLoader(web_paths=urls_to_load, requests_kwargs={"headers": headers})
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")


    # ------------------------------------------------------------------ #
    #            write manifest.json                                     #
    # ------------------------------------------------------------------ #
    additional_meta_list = []
    for doc in docs:
        item_url = doc.metadata.get("source", "")
        time.sleep(REQUEST_DELAY_SECONDS)
 
        additional_meta = _find_meta(item_url, headers)
 
        entry = {
            "published_date": additional_meta["published_date"],
            "license":        WHO_LICENSE,
        }

        additional_meta_list.append(entry)

        doc.metadata.update({k: v for k, v in entry.items() if v is not None})

    manifest = [doc.metadata for doc in docs]

    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest written to {manifest_path}")

    # for idx, doc in enumerate(docs):
    #         print(f"\n--- Document {idx + 1} ---")
    #         print("Source:", doc.metadata.get("source"))
    #         print("Content Preview:\n", doc.page_content[:300].strip())

    return docs
