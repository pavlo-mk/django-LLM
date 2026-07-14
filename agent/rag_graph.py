"""Dedicated retrieve→generate RAG pipeline (classic, non-agentic).

Unlike the agent — which *decides* whether to call the retriever tool — this
graph always retrieves first, then answers strictly from the retrieved context.
It's a plain two-node LangGraph: ``retrieve`` → ``generate``.
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TypedDict

import structlog
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .graph import build_llm
from .rag import asearch, format_docs

log = structlog.get_logger(__name__)

RAG_PROMPT = (
    "You are a question-answering assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, say you don't "
    "know — do not make anything up.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def build_rag_prompt(question: str, docs: list[Document]) -> str:
    return RAG_PROMPT.format(context=format_docs(docs), question=question)


class RagState(TypedDict):
    question: str
    context: list[Document]
    answer: str


async def _retrieve(state: RagState) -> dict:
    docs = await asearch(state["question"])
    log.info("rag.retrieve", question=state["question"], hits=len(docs))
    return {"context": docs}


async def _generate(state: RagState) -> dict:
    llm = build_llm()
    resp = await llm.ainvoke(build_rag_prompt(state["question"], state["context"]))
    return {"answer": resp.content}


@lru_cache(maxsize=1)
def get_rag_graph() -> CompiledStateGraph:
    graph = StateGraph(RagState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


async def arag_answer(question: str) -> dict:
    """Run the pipeline to completion; returns {answer, sources}."""
    result = await get_rag_graph().ainvoke({"question": question})
    sources = sorted({d.metadata.get("source", "?") for d in result["context"]})
    return {"answer": result["answer"], "sources": sources}


async def arag_stream(question: str) -> AsyncIterator[tuple[str, object]]:
    """Yield ("sources", [...]) once, then ("token", str) as the answer streams.

    Mirrors the graph's two steps (retrieve then generate) but streams the
    generation so the UI can render tokens live and show its sources up front.
    """
    docs = await asearch(question)
    log.info("rag.stream", question=question, hits=len(docs))
    yield "sources", sorted({d.metadata.get("source", "?") for d in docs})

    llm = build_llm()
    async for chunk in llm.astream(build_rag_prompt(question, docs)):
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
            yield "token", chunk.content
