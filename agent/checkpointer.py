"""LangGraph Postgres checkpointer.

The checkpointer is what gives the agent memory across HTTP requests: every
step of the graph is written to Postgres keyed by ``thread_id``, so a follow-up
request that reuses the same ``thread_id`` resumes the conversation.

We build a single process-wide ``PostgresSaver`` backed by a connection pool.
``setup()`` creates the checkpoint tables on first use (idempotent).
"""

import threading

from django.conf import settings
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_lock = threading.Lock()
_checkpointer: PostgresSaver | None = None
_pool: ConnectionPool | None = None


def get_checkpointer() -> PostgresSaver:
    """Return the shared PostgresSaver, creating it (and its tables) on first call."""
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    with _lock:
        if _checkpointer is None:
            _pool = ConnectionPool(
                conninfo=settings.CHECKPOINTER_DSN,
                max_size=10,
                open=True,
                kwargs={
                    # autocommit + no prepared statements is what PostgresSaver expects.
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            )
            checkpointer = PostgresSaver(_pool)
            checkpointer.setup()
            _checkpointer = checkpointer

    return _checkpointer
