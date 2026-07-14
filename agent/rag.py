"""Retrieval-Augmented Generation building blocks.

A single sync ``PGVector`` store (pgvector in the same Postgres) holds the
embedded document chunks. Embeddings come from Ollama — by default the same
``llama3.2`` model as the chat agent, swappable via ``OLLAMA_EMBED_MODEL``.

The store is used three ways:
- ``ingest_text`` — split + embed + upsert documents.
- ``search`` — sync similarity search, used by the retriever tool.
- ``asearch`` — async wrapper (``asyncio.to_thread``) for the async RAG graph.
"""

import asyncio
from functools import lru_cache

import structlog
from django.conf import settings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = structlog.get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    """The shared pgvector store; creates the extension + tables on first use."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.RAG_COLLECTION,
        connection=settings.VECTOR_DSN,
        use_jsonb=True,
        create_extension=True,
    )


def ingest_text(text: str, source: str) -> int:
    """Split ``text`` into chunks, embed, and store them. Returns chunk count."""
    chunks = _splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": source}) for c in chunks]
    get_vectorstore().add_documents(docs)
    log.info("rag.ingest", source=source, chunks=len(chunks))
    return len(chunks)


def search(query: str, k: int | None = None) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k or settings.RAG_TOP_K)


async def asearch(query: str, k: int | None = None) -> list[Document]:
    return await asyncio.to_thread(search, query, k)


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks as a single context string with source tags."""
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the ingested knowledge base for passages relevant to the query.

    Use this whenever the user asks about ingested documents, domain knowledge,
    or anything that isn't general knowledge or arithmetic.
    """
    return format_docs(search(query))
