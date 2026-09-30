"""Load WHO fact-sheet HTML, with an optional API loader."""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.data_loading.licensing import html_license

API_URL = "https://www.who.int/api/hubs/factsheets"
PAGE_ROOT = "https://www.who.int/news-room/fact-sheets/detail/"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "clinical-rag-portfolio-project/1.0 (personal, non-commercial use)",
}


def _date(item):
    for key in (
        "FormatedDate",
        "LocalPublicationDate",
        "PublicationDateAndTime",
        "PublicationDate",
    ):
        value = item.get(key)
        if not value:
            continue
        try:
            return (
                datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            )
        except ValueError:
            try:
                return datetime.strptime(value.strip(), "%d %B %Y").date().isoformat()
            except ValueError:
                pass
    return None


def _build_document(item):
    path = item.get("ItemDefaultUrl") or item.get("UrlName")
    if not path or not isinstance(item.get("Content"), str):
        raise ValueError("Fact sheet is missing its page URL or Content field.")
    source = (
        urljoin("https://www.who.int", path)
        if path.startswith(("http://", "https://", "/news-room/"))
        else urljoin(PAGE_ROOT, path.lstrip("/"))
    )
    if urlparse(source).hostname != "www.who.int" or not source.startswith(PAGE_ROOT):
        raise ValueError(f"Unexpected fact-sheet URL: {source}")
    soup = BeautifulSoup(item["Content"], "html.parser")
    rights = html_license(soup, source)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Retain table row relationships instead of flattening cells onto separate lines.
    for table in soup.find_all("table"):
        rows = [
            " | ".join(
                cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])
            )
            for row in table.find_all("tr")
        ]
        table.replace_with("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")
    return Document(
        page_content=soup.get_text("\n", strip=True),
        metadata={
            "document_id": hashlib.sha256(source.encode()).hexdigest()[:16],
            "source": source,
            "source_type": "web",
            "source_name": "WHO Fact Sheets",
            "title": item.get("Title"),
            "description": item.get("MetaDescription") or item.get("Summary"),
            "language": "en",
            "published_date": _date(item),
            "page_number": None,
            "n_pages": None,
            "content_origin": "api",
            **rights,
        },
    )


def save_manifest(docs, manifest_path):
    """Write a compact source inventory; full metadata stays on Documents."""
    entries = [
        {
            "title": d.metadata.get("title"),
            "item_url": d.metadata["source"],
            "pdf_url": None,
            "published_date": d.metadata.get("published_date"),
            "description": d.metadata.get("description"),
            "language": d.metadata.get("language"),
            "local_path": None,
            "license": d.metadata.get("license"),
        }
        for d in docs
    ]
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _request_batch(session, params, read_timeout, max_attempts):
    for attempt in range(1, max_attempts + 1):
        print(
            f"Fetching WHO API offset {params['$skip']} ({params['$top']} records), attempt {attempt}/{max_attempts}...",
            flush=True,
        )
        try:
            response = session.get(API_URL, params=params, timeout=(10, read_timeout))
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_attempts:
                raise requests.Timeout(
                    f"WHO API failed at offset {params['$skip']} after {max_attempts} attempts. "
                    "Try again later or use batch_size=1. The existing manifest was not replaced."
                ) from exc
            time.sleep(min(2**attempt, 8))


def load_who_fact_sheets_api(
    num_pages=None,
    manifest_path="data/who/manifest_web.json",
    headers=None,
    *,
    batch_size=5,
    read_timeout=60,
    max_attempts=3,
):
    """Fetch English records in stable URL order. None loads all; zero loads none."""
    if num_pages is not None and num_pages < 0:
        raise ValueError("num_pages must be nonnegative or None.")
    if batch_size < 1 or read_timeout <= 0 or max_attempts < 1:
        raise ValueError("batch_size, read_timeout and max_attempts must be positive.")
    docs, seen = [], set()
    offset = 0
    with requests.Session() as session:
        session.headers.update({**DEFAULT_HEADERS, **(headers or {})})
        while num_pages is None or len(docs) < num_pages:
            count = (
                min(batch_size, num_pages - len(docs))
                if num_pages is not None
                else batch_size
            )
            response = _request_batch(
                session,
                {
                    "$top": count,
                    "$skip": offset,
                    "$orderby": "UrlName",
                    "sf_culture": "en",
                },
                read_timeout,
                max_attempts,
            )
            payload = response.json()
            items = (
                payload
                if isinstance(payload, list)
                else payload.get("value")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(items, list):
                raise ValueError(
                    "Unexpected WHO API response: expected records in 'value'."
                )
            if not items:
                break
            new_count = 0
            for item in items:
                doc = _build_document(item)
                source = doc.metadata["source"]
                if source in seen:
                    continue
                seen.add(source)
                new_count += 1
                if doc.page_content.strip():
                    docs.append(doc)
                if num_pages is not None and len(docs) >= num_pages:
                    break
            if not new_count:
                raise ValueError(
                    "WHO API repeated records; pagination may not be supported."
                )
            offset += len(items)
            if num_pages is None or len(docs) < num_pages:
                time.sleep(1)
    if manifest_path:
        save_manifest(docs, manifest_path)
    print(f"Loaded {len(docs)} WHO fact sheets via API.")
    return docs


def load_who_fact_sheets(
    num_pages=None,
    manifest_path="data/who/manifest_web.json",
    headers=None,
    *,
    urls=None,
):
    """Load HTML pages; pass fixed URLs for a reproducible evaluation corpus."""
    if num_pages is not None and num_pages < 0:
        raise ValueError("num_pages must be nonnegative or None.")
    docs = []
    with requests.Session() as session:
        session.headers.update(
            {**DEFAULT_HEADERS, "Accept": "text/html", **(headers or {})}
        )
        if urls is None:
            index_url = "https://www.who.int/news-room/fact-sheets"
            response = session.get(index_url, timeout=(10, 30))
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            urls = sorted(
                {
                    urljoin(index_url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if urljoin(index_url, a["href"]).startswith(PAGE_ROOT)
                }
            )
            if not urls:
                raise ValueError("No fact-sheet links found on WHO's index page.")
        else:
            urls = list(dict.fromkeys(urls))
        if num_pages is not None:
            urls = urls[:num_pages]
        for i, url in enumerate(urls, 1):
            if not url.startswith(PAGE_ROOT):
                raise ValueError(f"Unexpected fact-sheet URL: {url}")
            print(f"Loading WHO HTML {i}/{len(urls)}: {url}", flush=True)
            response = session.get(url, timeout=(10, 30))
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            rights = html_license(soup, url)
            title = soup.find("h1") or soup.find("title")
            description = soup.find("meta", attrs={"name": "description"})
            date = soup.find("time")
            raw_date = (
                date.get("datetime") or date.get_text(" ", strip=True) if date else None
            )
            if not raw_date:
                date = soup.find(class_=re.compile("date", re.I))
                raw_date = date.get_text(" ", strip=True) if date else None
            language = soup.find("html")
            metadata = {
                "document_id": hashlib.sha256(url.encode()).hexdigest()[:16],
                "source": url,
                "source_type": "web",
                "source_name": "WHO Fact Sheets",
                "title": title.get_text(" ", strip=True) if title else None,
                "description": description.get("content") if description else None,
                "published_date": _date({"FormatedDate": raw_date}),
                "language": language.get("lang") if language else None,
                "page_number": None,
                "n_pages": None,
                "content_origin": "html",
                **rights,
            }
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = (soup.find("main") or soup).get_text("\n", strip=False)
            if not content.strip():
                raise ValueError(f"Empty fact-sheet content: {url}")
            docs.append(Document(page_content=content, metadata=metadata))
            if i < len(urls):
                time.sleep(1)
    if manifest_path:
        save_manifest(docs, manifest_path)
    print(f"Loaded {len(docs)} WHO fact sheets via HTML.")
    return docs
