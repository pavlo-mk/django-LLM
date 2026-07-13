"""Toy tools exposed to the LangGraph agent.

Each tool is a plain function decorated with ``@tool``; LangChain derives the
name, description, and argument schema from the signature and docstring, and
LangGraph lets the model call them by name.
"""

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return their product."""
    return a * b


@tool
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in a piece of text."""
    return len(text.split())


TOOLS = [add, multiply, current_time, word_count]
