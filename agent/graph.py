"""The LangGraph agent.

We use LangGraph's prebuilt ReAct agent: a ``StateGraph`` with two nodes — the
LLM (which may emit tool calls) and a tool executor — looping until the model
answers without calling a tool. The model is served locally by Ollama.

Two entry points share the same graph definition and Postgres-backed memory:

* :func:`run` — synchronous, used by the blocking ``/api/chat/`` endpoint.
* :func:`astream_tokens` — asynchronous token stream, used by the SSE endpoint
  under ASGI so many streams run concurrently without a thread each.
"""

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

import structlog
from django.conf import settings
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .checkpointer import get_async_checkpointer, get_checkpointer
from .tools import TOOLS

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant running inside a Django demo app. "
    "You have tools for arithmetic, the current time, and counting words. "
    "Use a tool whenever it gives a more accurate answer than guessing, and "
    "otherwise answer directly."
)


def _build_llm() -> ChatOllama:
    # client_kwargs are forwarded to the underlying httpx client, giving us a
    # per-request timeout so a hung model can't hang the web worker forever.
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        client_kwargs={"timeout": settings.OLLAMA_TIMEOUT},
    )


@lru_cache(maxsize=1)
def get_agent() -> CompiledStateGraph:
    """Build (once) and return the compiled sync ReAct agent."""
    return create_react_agent(
        _build_llm(),
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
    )


async def get_async_agent() -> CompiledStateGraph:
    """Build (once) and return the compiled async ReAct agent."""
    global _async_agent
    if _async_agent is None:
        _async_agent = create_react_agent(
            _build_llm(),
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=await get_async_checkpointer(),
        )
    return _async_agent


_async_agent: CompiledStateGraph | None = None


def run(thread_id: str, message: str) -> str:
    """Run the agent to completion for one user turn and return the reply text."""
    agent = get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    log.info("agent.run", thread_id=thread_id)
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
    return result["messages"][-1].content


def stream_tokens(thread_id: str, message: str) -> Iterator[str]:
    """Synchronous token stream (kept for scripts/tests)."""
    agent = get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        content = _token(chunk)
        if content:
            yield content


async def astream_tokens(thread_id: str, message: str) -> AsyncIterator[str]:
    """Asynchronously yield assistant token strings as the agent produces them.

    Uses ``stream_mode="messages"`` which emits ``(chunk, metadata)`` for every
    message the graph produces. We forward only the model's own tokens
    (``AIMessageChunk``) so raw tool results never leak into the reply.
    """
    agent = await get_async_agent()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    log.info("agent.astream", thread_id=thread_id)
    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        content = _token(chunk)
        if content:
            yield content


def _token(chunk: object) -> str:
    """Extract a text token from a stream chunk, or '' if it isn't model text."""
    if not isinstance(chunk, AIMessageChunk):
        return ""
    content = chunk.content
    return content if isinstance(content, str) else ""
