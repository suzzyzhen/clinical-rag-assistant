"""
schema.py
Typed data structures shared across the ETL pipeline.

Keeping these as dataclasses (not raw dicts) means every stage of the
pipeline — cleaning, chunking, metadata, versioning — agrees on the same
shape, and it's trivial to serialize to JSONL for the embedding stage.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    GUIDELINE = "clinical_guideline"
    LITERATURE = "fact_sheets"
    POLICY = "institutional_policy"
    CLINICAL_NOTE = "clinical_note"  


class SourceLicense(str, Enum):
    """Track licensing explicitly -- important for a clinical RAG demo,
    since it shows you thought about provenance and reuse rights."""
    PUBLIC_DOMAIN = "public_domain"
    CC_BY = "cc_by"
    CC_BY_NC = "CC BY-NC-SA 3.0 IGO"
    UNKNOWN = "unknown"


# @dataclass
# class RawDocument:
#     """A single source document before cleaning/chunking."""
#     doc_id: str
#     page_content: str
#     metadata: dict              # mirrors LangChain's flat metadata dict --
#                                 # keys: source, title, description, language,
#                                 # doc_type, source_name, license,
#                                 # published_date, and any extras (n_pages etc.)
#     retrieved_at: str = field(
#         default_factory=lambda: datetime.now(timezone.utc).isoformat()
#     )

#     def content_hash(self) -> str:
#         return hashlib.sha256(self.page_content.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """A single retrievable unit that will get embedded."""
    chunk_id: str               # f"{doc_id}::chunk::{index}"
    doc_id: str
    text: str
    chunk_index: int
    n_tokens_approx: int
    section_title: Optional[str]
    metadata: dict              # flattened, embedding/store-friendly metadata
    version: int                 # which ingest version of the doc this came from
    content_hash: str            # hash of doc content this chunk was derived from

    def to_dict(self) -> dict:
        return asdict(self)