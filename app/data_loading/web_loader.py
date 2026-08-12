import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


WHO_LICENSE = "CC BY-NC-SA 3.0 IGO"
DEFAULT_HEADERS = {
    "User-Agent": "clinical-rag-portfolio-project/1.0 (personal, non-commercial use)"
}
REQUEST_DELAY_SECONDS = 1.0


def _make_document_id(source: str) -> str:
    """Create a stable document ID from the source URL."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _normalize_date(date_string: Optional[str]) -> Optional[str]:
    """
    Normalize a date to YYYY-MM-DD.

    Examples:
        2025-10-15 -> 2025-10-15
        15 October 2025 -> 2025-10-15
    """
    if not date_string:
        return None

    date_string = date_string.strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_string):
        return date_string

    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", date_string)
    if not match:
        return None

    day, month, year = match.groups()
    months = {
        "january": "01", "february": "02", "march": "03",
        "april": "04",   "may": "05",      "june": "06",
        "july": "07",    "august": "08",   "september": "09",
        "october": "10", "november": "11", "december": "12",
    }
    month_number = months.get(month.lower())

    if not month_number:
        return None

    return f"{year}-{month_number}-{int(day):02d}"


def _extract_published_date(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract publication date from a WHO fact-sheet page.
    Resolution order:
    1. <meta name="citation_date">
    2. <meta name="dc.date">
    3. <meta name="date">
    4. First element with a class containing "date"
    5. Regex scan of the page text for "DD Month YYYY"
    """
    _DATE_PATTERN = re.compile(r"\b\d{1,2} \w+ \d{4}\b")

    # 1. Metadata tags
    for meta_name in ("citation_date", "dc.date", "date"):
        meta = soup.find("meta", attrs={"name": meta_name})
        if meta and meta.get("content"):
            normalized = _normalize_date(meta["content"].strip())
            if normalized:
                return normalized

    # 2. Element whose class contains "date"
    date_el = soup.find(class_=re.compile(r"date", re.IGNORECASE))
    if date_el:
        match = _DATE_PATTERN.search(date_el.get_text(strip=True))
        if match:
            normalized = _normalize_date(match.group(0))
            if normalized:
                return normalized

    # 3. Fallback: scan full page text
    match = _DATE_PATTERN.search(soup.get_text())
    if match:
        normalized = _normalize_date(match.group(0))
        if normalized:
            return normalized

    return None


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    """Extract the webpage title."""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return title

    html_title = soup.find("title")
    if html_title:
        title = html_title.get_text(" ", strip=True)
        if title:
            return title

    return None


def _extract_description(soup: BeautifulSoup) -> Optional[str]:
    """Extract the webpage meta description."""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def _extract_language(soup: BeautifulSoup) -> Optional[str]:
    """Extract the HTML language attribute."""
    html = soup.find("html")
    if html and html.get("lang"):
        return html["lang"].strip()
    return None


def _extract_raw_text(soup: BeautifulSoup) -> str:
    """
    Extract raw textual content from the webpage.

    This function intentionally does minimal processing.
    Cleaning such as navigation removal, whitespace normalization,
    and boilerplate removal happens later in cleaning.py.
    """
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if main:
        return main.get_text("\n", strip=False)

    return soup.get_text("\n", strip=False)


def _build_document(
    url: str,
    soup: BeautifulSoup,
    source_name: str,
    license: str,
) -> Document:
    """Build a normalized LangChain Document."""
    metadata = {
        "document_id":    _make_document_id(url),
        "source":         url,
        "source_type":    "web",
        "source_name":    source_name,
        "title":          _extract_title(soup),
        "description":    _extract_description(soup),
        "language":       _extract_language(soup),
        "published_date": _extract_published_date(soup),
        "license":        license,
        "page_number":    None,
        "n_pages":        None,
    }
    return Document(page_content=_extract_raw_text(soup), metadata=metadata)


def load_who_fact_sheets(
    source_url: str = "https://www.who.int/news-room/fact-sheets",
    target_prefix: str = "https://www.who.int/news-room/fact-sheets/detail/",
    headers: Optional[dict] = None,
    manifest_path: Optional[str] = "data/who/manifest_web.json",
    num_pages: Optional[int] = None,
    license: str = WHO_LICENSE,
) -> list[Document]:
    """
    Load WHO fact sheets into LangChain Documents.

    Each webpage becomes one Document.
    No text cleaning or chunking is performed here.
    """
    headers = headers or DEFAULT_HEADERS

    print(f"Fetching WHO fact-sheet index: {source_url}")
    response = requests.get(source_url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # --- Find fact-sheet URLs ---
    urls = set()
    for anchor in soup.find_all("a", href=True):
        full_url = urljoin(source_url, anchor["href"])
        if full_url.startswith(target_prefix):
            urls.add(full_url)

    urls_to_load = sorted(urls)
    print(f"Found {len(urls_to_load)} fact-sheet URLs.")

    # --- Optional sampling for development ---
    if num_pages is not None and num_pages > 0:
        sample_size = min(num_pages, len(urls_to_load))
        urls_to_load = random.sample(urls_to_load, sample_size)

    print(f"Loading {len(urls_to_load)} WHO fact sheets.")

    # --- Load pages ---
    docs = []
    for index, url in enumerate(urls_to_load, start=1):
        print(f"[{index}/{len(urls_to_load)}] {url}")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            doc = _build_document(url=url, soup=soup, source_name="WHO Fact Sheets", license=license)

            if not doc.page_content.strip():
                print("  ! Empty page. Skipping.")
                continue

            docs.append(doc)

        except requests.RequestException as e:
            print(f"  ! Request failed: {e}")
        except Exception as e:
            print(f"  ! Failed to process: {e}")

        if index < len(urls_to_load):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Successfully loaded {len(docs)} documents.")

    if manifest_path:
        save_manifest(docs, manifest_path)

    return docs


def save_manifest(docs: list[Document], manifest_path: str) -> None:
    """Save document metadata to a JSON manifest."""
    manifest = [doc.metadata for doc in docs]
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {path}")