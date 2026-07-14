"""Document ingestion: read a source, embed it into the RAG store, and record it.

Shared by the ``ingest`` management command (bulk seeding) and the upload view.
"""

from pathlib import Path
from typing import BinaryIO

from agent.rag import ingest_text

from .models import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def read_path(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with path.open("rb") as fp:
            return _read_pdf(fp)
    return path.read_text(encoding="utf-8", errors="ignore")


def read_upload(uploaded) -> str:
    """Read text from a Django UploadedFile (supports .pdf and text formats)."""
    if (uploaded.name or "").lower().endswith(".pdf"):
        return _read_pdf(uploaded)
    return uploaded.read().decode("utf-8", errors="ignore")


def _read_pdf(fp: BinaryIO) -> str:
    from pypdf import PdfReader  # imported lazily so non-PDF paths stay light

    reader = PdfReader(fp)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ingest(text: str, source: str) -> Document:
    """Embed ``text`` into the vector store and record a Document row."""
    chunks = ingest_text(text, source=source)
    return Document.objects.create(source=source, char_count=len(text), chunk_count=chunks)
