import json
import re
import time
from pathlib import Path

import requests

IRIS_API = "https://iris.who.int/server/api"
WER_COLLECTION_HANDLE = "10665/2650"   # WHO WER collection handle on IRIS
HEADERS = {
    "User-Agent": "clinical-rag-portfolio-project/1.0 (personal, non-commercial use)",
    "Accept": "application/json",
}
REQUEST_DELAY_SECONDS = 1.0
WHO_LICENSE = "CC BY-NC-SA 3.0 IGO"


def _resolve_handle_to_uuid(handle: str) -> str:
    """Resolve a DSpace handle (e.g. 10665/2650) to its internal UUID
    via the IRIS REST API -- needed to query the collection's items."""
    resp = requests.get(
        f"{IRIS_API}/core/handles/{handle}",
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # The response contains a self link like .../collections/<uuid>
    self_href = data["_links"]["self"]["href"]
    return self_href.rstrip("/").split("/")[-1]


def _get_recent_wer_items(count: int) -> list[dict]:
    resp = requests.get(
        f"{IRIS_API}/discover/search/objects",
        headers=HEADERS,
        params={
            "query":   "dc.title:\"Weekly Epidemiological Record\"",
            "dsoType": "item",
            "sort":    "dc.date.issued,DESC",
            "page":    0,
            "size":    count,
            "embed":   "bundles/bitstreams",
            "f.title": "Weekly Epidemiological Record,contains",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get("_embedded", {})
            .get("searchResult", {})
            .get("_embedded", {})
            .get("objects", [])
    )


def _get_recent_who_drug_information(count: int) -> list[dict]:
    resp = requests.get(
        f"{IRIS_API}/discover/search/objects",
        headers=HEADERS,
        params={
            "query":   "dc.title:\"WHO Drug Information\"",
            "dsoType": "item",
            "sort":    "dc.date.issued,DESC",
            "page":    0,
            "size":    count * 10,
            "embed":   "bundles/bitstreams",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    all_items = (
        data.get("_embedded", {})
            .get("searchResult", {})
            .get("_embedded", {})
            .get("objects", [])
    )

    drug_info_items = [
        item for item in all_items
        if "who drug information" in (
            item.get("_embedded", {})
                .get("indexableObject", {})
                .get("metadata", {})
                .get("dc.title", [{}])[0]
                .get("value", "")
                .lower()
        )
    ]

    return drug_info_items[:count]


def _extract_english_pdf_url(item: dict) -> tuple[str | None, str | None]:
    """Walks the embedded bundle/bitstream tree for the English PDF.
    WER PDFs follow the naming pattern WER<vol><issue>-eng-fre.pdf.
    Returns (pdf_url, filename) or (None, None) if not found."""
    bundles = (
        item.get("_embedded", {})
            .get("indexableObject", {})
            .get("_embedded", {})
            .get("bundles", {})
            .get("_embedded", {})
            .get("bundles", [])
    )
    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        bitstreams = (
            bundle.get("_embedded", {})
                  .get("bitstreams", {})
                  .get("_embedded", {})
                  .get("bitstreams", [])
        )
        for bs in bitstreams:
            name = bs.get("name", "")
            # prefer the bilingual eng-fre PDF; fall back to any -eng PDF
            if re.search(r"-eng", name, re.IGNORECASE) and name.endswith(".pdf"):
                content_url = (
                    bs.get("_links", {})
                      .get("content", {})
                      .get("href")
                )
                return content_url, name

    return None, None


def _extract_metadata(item: dict) -> dict:
    """Pulls title and issue date out of the DSpace metadata dict."""
    obj = item.get("_embedded", {}).get("indexableObject", {})
    metadata = obj.get("metadata", {})

    def first_value(field: str):
        values = metadata.get(field, [])
        return values[0]["value"] if values else None

    return {
        "title":          first_value("dc.title"),
        "published_date": first_value("dc.date.issued"),
        "handle":         obj.get("handle"),
    }


def _make_drug_info_filename(title: str) -> str | None:
    """
    Parses a WHO Drug Information title into a clean filename.
    """
    match = re.search(
        r"WHO Drug Information\s+(\d{4}),\s*vol\.\s*(\d+),\s*(\d+)",
        title,
        re.IGNORECASE,
    )
    if not match:
        return None

    year, volume, issue = match.groups()
    return f"WHO_drug_information_{year}_v{volume}_{issue}.pdf"


def download_recent_publications(wer_count: int = 5, drug_info_count: int = 2, out_dir: str = "../../data/who") -> list[dict]:
    """
    Downloads the `count` most recent WER issues from IRIS as English PDFs.
    Returns a manifest list matching the project's standard schema.

    Usage:
        manifest = download_recent_wer(count=2, out_dir="data/who")
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {wer_count} most recent WER items from IRIS...")
    wer_items = _get_recent_wer_items(wer_count)

    print(f"Fetching {drug_info_count} most recent WHO Drug Information items from IRIS...")
    drug_items = _get_recent_who_drug_information(drug_info_count)

    all_items = wer_items + drug_items
    print(f"Total items to download: {len(all_items)}")

    manifest = []
    for i, item in enumerate(all_items, start=1):
        meta = _extract_metadata(item)
        pdf_url, filename = _extract_english_pdf_url(item)
        # rename WHO Drug Information pdfs
        if "who drug information" in (meta.get("title") or "").lower():
            renamed = _make_drug_info_filename(meta.get("title", ""))
            if renamed:
                filename = renamed

        item_url = f"https://iris.who.int/handle/{meta['handle']}" if meta.get("handle") else None

        print(f"[{i}/{len(all_items)}] {meta.get('title')}")

        entry = {
            "title":          meta.get("title"),
            "item_url":       item_url,
            "pdf_url":        pdf_url,
            "published_date": meta.get("published_date"),
            "local_path":     None,
            "license":        WHO_LICENSE,
        }

        if not pdf_url:
            print("  ! No English PDF found, skipping download.")
            manifest.append(entry)
            continue

        local_path = out_path / filename
        print(f"  -> Downloading: {filename}")
        try:
            resp = requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            entry["local_path"] = str(local_path)
            print(f"  -> Saved to: {local_path}")
        except requests.RequestException as e:
            print(f"  ! Download failed: {e}")

        manifest.append(entry)
        if i < len(all_items):
            time.sleep(REQUEST_DELAY_SECONDS)

    manifest_path = out_path / "manifest_wer.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")

    return manifest