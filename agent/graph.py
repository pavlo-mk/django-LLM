"""The LangGraph agent.

We use LangGraph's prebuilt ReAct agent: a ``StateGraph`` with two nodes — the
LLM (which may emit tool calls) and a tool executor — looping until the model
answers without calling a tool. The model is served locally by Ollama.
"""

from collections.abc import Iterator
from functools import lru_cache

from django.conf import settings
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .checkpointer import get_checkpointer
from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant running inside a Django demo app. "
    "You have tools for arithmetic, the current time, and counting words. "
    "Use a tool whenever it gives a more accurate answer than guessing, and "
    "otherwise answer directly."
)


@lru_cache(maxsize=1)
def get_agent() -> CompiledStateGraph:
    """Build (once) and return the compiled ReAct agent."""
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )
    return create_react_agent(
        llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
    )


def run(thread_id: str, message: str) -> str:
    """Run the agent to completion for one user turn and return the reply text."""
    agent = get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
    return result["messages"][-1].content


def stream_tokens(thread_id: str, message: str) -> Iterator[str]:
    """Yield assistant token strings as the agent produces them.

    Uses ``stream_mode="messages"`` which emits ``(chunk, metadata)`` for every
    message the graph produces. We forward only the model's own tokens
    (``AIMessageChunk``) so raw tool results never leak into the reply.
    """
    agent = get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        content = chunk.content
        if isinstance(content, str) and content:
            yield content
