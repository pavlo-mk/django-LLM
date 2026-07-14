"""LangGraph Postgres checkpointer.

The checkpointer is what gives the agent memory across HTTP requests: every
step of the graph is written to Postgres keyed by ``thread_id``, so a follow-up
request that reuses the same ``thread_id`` resumes the conversation.

Two savers share the same checkpoint tables:

* a sync ``PostgresSaver`` used by the blocking ``/api/chat/`` endpoint, and
* an async ``AsyncPostgresSaver`` used by the streaming endpoint (under ASGI).

Each is a process-wide singleton; ``setup()`` creates the tables on first use
(idempotent), so whichever runs first wins and the other is a no-op.
"""

import asyncio
import threading

from django.conf import settings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

# Passed to every pooled connection: autocommit + no prepared statements is
# what the LangGraph Postgres savers expect, with dict rows.
_CONN_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

_lock = threading.Lock()
_checkpointer: PostgresSaver | None = None
_pool: ConnectionPool | None = None

_async_lock = asyncio.Lock()
_async_checkpointer: AsyncPostgresSaver | None = None
_async_pool: AsyncConnectionPool | None = None


def get_checkpointer() -> PostgresSaver:
    """Return the shared sync PostgresSaver, creating it (and its tables) on first call."""
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    with _lock:
        if _checkpointer is None:
            _pool = ConnectionPool(
                conninfo=settings.CHECKPOINTER_DSN,
                max_size=10,
                open=True,
                kwargs=_CONN_KWARGS,
            )
            # row_factory=dict_row is set via kwargs above, but the pool's
            # generic type still reads as tuple-rows to the type checker.
            checkpointer = PostgresSaver(_pool)  # type: ignore[arg-type]
            checkpointer.setup()
            _checkpointer = checkpointer

    return _checkpointer


async def get_async_checkpointer() -> AsyncPostgresSaver:
    """Return the shared AsyncPostgresSaver, opening its pool in the running loop."""
    global _async_checkpointer, _async_pool
    if _async_checkpointer is not None:
        return _async_checkpointer

    async with _async_lock:
        if _async_checkpointer is None:
            pool = AsyncConnectionPool(
                conninfo=settings.CHECKPOINTER_DSN,
                max_size=10,
                open=False,  # must be opened inside the event loop
                kwargs=_CONN_KWARGS,
            )
            await pool.open()
            checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
            await checkpointer.setup()
            _async_pool = pool
            _async_checkpointer = checkpointer

    return _async_checkpointer
