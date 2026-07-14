# dj-llm — Django + LangGraph agent

[![CI](https://github.com/pavlo-mk/django-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/pavlo-mk/django-LLM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pavlo-mk/django-LLM/graph/badge.svg)](https://codecov.io/gh/pavlo-mk/django-LLM)

A small but complete test project that wires a **Django 6** web app to a
**LangGraph** ReAct agent running on a **local Ollama** model. Conversation
state is persisted in **Postgres** using LangGraph's Postgres checkpointer, so
the agent remembers a conversation across HTTP requests. It also does **RAG**
over **pgvector** (same Postgres) — both as a retriever *tool* the agent can
call and as a dedicated retrieve→generate pipeline. The UI supports a blocking
JSON endpoint and **async token streaming over SSE** (served under ASGI). The
whole stack runs in Docker, built with **uv**.

```
Browser ─HTTP─▶ Django (ASGI/uvicorn) ─▶ agent.graph (LangGraph ReAct)
                     │                        │        │
                 Postgres                 ChatOllama  tools ── search_knowledge_base ─┐
              (ORM + checkpointer      (local model)  (math, time, words)             │
               + pgvector store)            │                                         ▼
                     ▲                       └──▶ rag_graph (retrieve→generate) ──▶ pgvector
                     └──────────────── OllamaEmbeddings (nomic-embed-text) ──────────┘
```

## Stack

| Piece            | Choice                                                  |
|------------------|---------------------------------------------------------|
| Web framework    | Django 6.0, served via ASGI (uvicorn)                   |
| Agent runtime    | LangGraph (prebuilt ReAct agent + a custom RAG graph)   |
| LLM              | Ollama `llama3.2` (local, tool-calling capable)         |
| RAG              | pgvector + `langchain-postgres`, Ollama embeddings      |
| Persistence      | Postgres 16 / pgvector (ORM + checkpointer + vectors)   |
| Streaming        | Async Server-Sent Events via `StreamingHttpResponse`    |
| Config           | Typed `pydantic-settings`                               |
| Static files     | WhiteNoise                                              |
| Observability    | structlog + optional Sentry / LangSmith                 |
| Packaging / dev  | uv, ruff, mypy, pytest, pre-commit, Docker              |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`brew install uv`) — manages Python + deps
- Docker (for Postgres)
- [Ollama](https://ollama.com) installed natively (`brew install ollama` on macOS —
  native gives Metal GPU acceleration; Docker Ollama is CPU-only on Mac)

uv installs a matching Python (3.12+) itself, so you don't need one preinstalled.

## Setup

```bash
# 1. Python + deps (creates .venv from uv.lock)
uv sync

# 2. Environment
cp .env.example .env          # tweak if the default ports clash

# 3. Postgres only (host port 5433 by default to avoid clashing with a local 5432)
docker compose up -d db

# 4. Ollama: start the server and pull the chat + embedding models
ollama serve &                # or run the Ollama.app
ollama pull llama3.2          # chat / tool-calling
ollama pull nomic-embed-text  # RAG embeddings (chat models can't always embed)

# 5. Django — serve under ASGI so async streaming works
uv run python manage.py migrate
uv run python manage.py createsuperuser        # optional, for /admin
uv run python manage.py ingest sample_docs/    # seed the RAG knowledge base
uv run uvicorn config.asgi:application --reload
```

Prefix commands with `uv run`, or activate the env once with
`source .venv/bin/activate`. There's also a `Makefile` — `make install`,
`make run`, `make test`, `make check`, `make up`.

> **Note:** use `uvicorn` (ASGI), not `manage.py runserver` (WSGI) — the async
> SSE streaming endpoint requires an ASGI server.

Open http://127.0.0.1:8000/ and chat. Toggle **stream** to switch between
streaming and blocking responses. The Django admin at `/admin/` shows the
`Thread` and `Message` records.

## Run everything in Docker

Build and run the app + Postgres together (the app image is a multi-stage
uv build; Ollama still runs natively on the host and is reached via
`host.docker.internal`):

```bash
docker compose up -d --build      # or: make up
```

The `app` service runs migrations + `collectstatic` on start, then serves via
uvicorn on http://localhost:8000/ with a `/healthz/` healthcheck. The `db`
service uses the `pgvector/pgvector` image so RAG works out of the box. Seed the
knowledge base with `docker compose exec app python manage.py ingest sample_docs/`.

## How it fits together

- **`agent/tools.py`** — plain functions (`add`, `multiply`, `current_time`,
  `word_count`) decorated with `@tool`.
- **`agent/graph.py`** — builds the ReAct agent with `create_react_agent`,
  a `ChatOllama` model (with a request timeout), the tools, and the
  checkpointer. Exposes `run()` (blocking) and `astream_tokens()` (async;
  yields the model's `AIMessageChunk` tokens only, so raw tool output never
  leaks into the reply).
- **`agent/checkpointer.py`** — process-wide sync `PostgresSaver` and async
  `AsyncPostgresSaver`, sharing the same tables; `thread_id` is the key.
- **`agent/rag.py`** — the pgvector store + Ollama embeddings, `ingest_text`,
  `search`/`asearch`, and the `search_knowledge_base` retriever tool.
- **`agent/rag_graph.py`** — the dedicated retrieve→generate LangGraph pipeline
  (`arag_answer`, `arag_stream`).
- **`chat/ingestion.py`** + **`chat/management/commands/ingest.py`** — read
  files (txt/md/pdf), embed, and record a `Document`.
- **`config/appconfig.py`** — typed `pydantic-settings` model; all env config
  lives here.
- **`chat/models.py`** — `Thread` / `Message` (conversation) and `Document`
  (ingested sources), mirrored into the ORM for the admin and UI.
- **`chat/views.py`** — `create_thread`, `chat` (blocking JSON, sync),
  `stream` (async agent SSE), `rag_stream` (async RAG SSE), `ingest_document`,
  `thread_messages`, and `healthz`.
- **`templates/chat/index.html`** — a dependency-free chat page (`fetch` +
  `EventSource`) with an Agent/RAG mode toggle and a document uploader.

## RAG

Two ways to use the knowledge base, both over the same pgvector store:

1. **Agentic (tool):** in *Agent* mode the ReAct agent calls
   `search_knowledge_base` when it decides retrieval helps — mixed freely with
   the other tools.
2. **Pipeline (graph):** in *RAG* mode a dedicated `retrieve → generate` graph
   always retrieves first and answers strictly from context, streaming the
   answer after emitting a `sources` event.

Ingest documents via the CLI (`python manage.py ingest sample_docs/` or any
file/folder of `.txt`/`.md`/`.pdf`) or the **＋ doc** uploader in the UI.
Embeddings use a dedicated model (`OLLAMA_EMBED_MODEL`, default
`nomic-embed-text`) because chat models like `llama3.2` can't serve embeddings
on every Ollama build.

## Endpoints

| Method | Path                                       | Purpose                       |
|--------|--------------------------------------------|-------------------------------|
| GET    | `/`                                        | Chat UI                       |
| GET    | `/healthz/`                                | Liveness probe                |
| POST   | `/api/threads/`                            | Create a thread               |
| GET    | `/api/threads/<uuid>/messages/`            | List a thread's messages      |
| POST   | `/api/chat/`                               | Blocking agent turn (JSON)    |
| GET    | `/api/threads/<uuid>/stream/?message=...`  | Async agent turn (SSE)        |
| GET    | `/api/threads/<uuid>/rag/?message=...`     | Async RAG pipeline turn (SSE) |
| POST   | `/api/ingest/`                             | Ingest a file or text         |

## Tests

Tests run under **pytest** (with `pytest-django`); agent/LLM calls are mocked so
they're fast and offline.

```bash
uv run pytest                                  # or: make test
RUN_AGENT_TESTS=1 uv run pytest chat/tests.py::AgentSmokeTest   # live Ollama
```

## Code quality

Linting/formatting is [ruff](https://docs.astral.sh/ruff/) and type checking is
[mypy](https://mypy-lang.org/); both are configured in `pyproject.toml`.
Install the git hooks with `uv run pre-commit install` (or `make install`).

```bash
uv run ruff check .           # lint (PEP 8 / pyflakes / import order / pyupgrade)
uv run ruff format .          # auto-format (add --check to only verify)
uv run mypy .                 # static type check
uv run pre-commit run --all-files
```

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request to `main` (and on pushes
to `main`). It spins up a Postgres service, installs deps with `uv sync
--frozen`, then runs, in order: `ruff check`, `ruff format --check`, `mypy`,
`manage.py check`, and `pytest` (with coverage uploaded to Codecov). The live
Ollama test is skipped in CI — it only runs locally with `RUN_AGENT_TESTS=1`.
Dependabot (`.github/dependabot.yml`) keeps uv, GitHub Actions, and the Docker
base image up to date.

## Configuration

All config is a typed `pydantic-settings` model in `config/appconfig.py`, read
from the environment and a local `.env` (see `.env.example`): Django
secret/debug/hosts, Postgres connection, `OLLAMA_BASE_URL` / `OLLAMA_MODEL` /
`OLLAMA_TIMEOUT`, an optional `CHECKPOINTER_DSN`, and observability toggles
(`SENTRY_DSN`, and LangSmith via `LANGSMITH_TRACING` / `LANGSMITH_API_KEY`).

## Notes & ideas to extend

- Swap `llama3.2` for `qwen2.5` (stronger tool use) by setting `OLLAMA_MODEL`.
- Add real tools (web search, DB lookups over your Django models).
- Add per-user threads by attaching `Thread` to `request.user`.
- Add LangSmith tracing (set the env vars) to inspect each agent step.

<img width="722" height="490" alt="image" src="https://github.com/user-attachments/assets/69ce473b-41c4-45bf-a282-4626bb4a71e0" />

