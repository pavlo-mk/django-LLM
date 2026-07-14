# dj-llm — Django + LangGraph agent

[![CI](https://github.com/pavlo-mk/django-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/pavlo-mk/django-LLM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pavlo-mk/django-LLM/graph/badge.svg)](https://codecov.io/gh/pavlo-mk/django-LLM)

A small but complete test project that wires a **Django 6** web app to a
**LangGraph** ReAct agent running on a **local Ollama** model. Conversation
state is persisted in **Postgres** using LangGraph's Postgres checkpointer, so
the agent remembers a conversation across HTTP requests. The UI supports a
blocking JSON endpoint and **async token streaming over SSE** (served under
ASGI). The whole stack runs in Docker, built with **uv**.

```
Browser ──HTTP──▶ Django (ASGI/uvicorn) ──▶ agent.graph (LangGraph ReAct)
                       │                          │            │
                   Postgres                   ChatOllama    tools
                (Django ORM +               (local model)  (add, multiply,
                 checkpointer)                             time, word_count)
```

## Stack

| Piece            | Choice                                                  |
|------------------|---------------------------------------------------------|
| Web framework    | Django 6.0, served via ASGI (uvicorn)                   |
| Agent runtime    | LangGraph (prebuilt ReAct agent)                        |
| LLM              | Ollama `llama3.2` (local, tool-calling capable)         |
| Persistence      | Postgres 16 (Django ORM + LangGraph checkpointer)       |
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

# 4. Ollama: start the server and pull the model
ollama serve &                # or run the Ollama.app
ollama pull llama3.2

# 5. Django — serve under ASGI so async streaming works
uv run python manage.py migrate
uv run python manage.py createsuperuser        # optional, for /admin
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
uvicorn on http://localhost:8000/ with a `/healthz/` healthcheck.

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
- **`config/appconfig.py`** — typed `pydantic-settings` model; all env config
  lives here.
- **`chat/models.py`** — `Thread` / `Message` mirror the conversation into the
  ORM for the admin and UI. The `thread_id` matches the checkpointer key.
- **`chat/views.py`** — `create_thread`, `chat` (blocking JSON, sync),
  `stream` (async SSE), `thread_messages`, and `healthz`.
- **`templates/chat/index.html`** — a dependency-free chat page using `fetch`
  and `EventSource`.

## Endpoints

| Method | Path                                       | Purpose                     |
|--------|--------------------------------------------|-----------------------------|
| GET    | `/`                                        | Chat UI                     |
| GET    | `/healthz/`                                | Liveness probe              |
| POST   | `/api/threads/`                            | Create a thread             |
| GET    | `/api/threads/<uuid>/messages/`            | List a thread's messages    |
| POST   | `/api/chat/`                               | Blocking turn (JSON reply)  |
| GET    | `/api/threads/<uuid>/stream/?message=...`  | Async streaming turn (SSE)  |

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

<img width="756" height="643" alt="image" src="https://github.com/user-attachments/assets/e44e60aa-4400-4522-9be7-1bc6723ed377" />

