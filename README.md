# dj-llm — Django + LangGraph agent

A small but complete test project that wires a **Django 6** web app to a
**LangGraph** ReAct agent running on a **local Ollama** model. Conversation
state is persisted in **Postgres** (via Docker) using LangGraph's Postgres
checkpointer, so the agent remembers a conversation across HTTP requests. The
UI supports both a blocking JSON endpoint and **token streaming over SSE**.

```
Browser ──HTTP──▶ Django views ──▶ agent.graph (LangGraph ReAct)
                       │                 │            │
                   Postgres          ChatOllama    tools
                (Django ORM +      (local model)  (add, multiply,
                 checkpointer)                     time, word_count)
```

## Stack

| Piece            | Choice                                              |
|------------------|-----------------------------------------------------|
| Web framework    | Django 6.0                                          |
| Agent runtime    | LangGraph (prebuilt ReAct agent)                    |
| LLM              | Ollama `llama3.2` (local, tool-calling capable)     |
| Persistence      | Postgres 16 in Docker (Django ORM + checkpointer)   |
| Streaming        | Server-Sent Events via `StreamingHttpResponse`      |

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

# 3. Postgres (host port 5433 by default to avoid clashing with a local 5432)
docker compose up -d

# 4. Ollama: start the server and pull the model
ollama serve &                # or run the Ollama.app
ollama pull llama3.2

# 5. Django
uv run python manage.py migrate
uv run python manage.py createsuperuser   # optional, for /admin
uv run python manage.py runserver
```

Prefix Django/tooling commands with `uv run`, or activate the env once with
`source .venv/bin/activate`.

Open http://127.0.0.1:8000/ and chat. Toggle **stream** to switch between
streaming and blocking responses. The Django admin at `/admin/` shows the
`Thread` and `Message` records.

## How it fits together

- **`agent/tools.py`** — plain functions (`add`, `multiply`, `current_time`,
  `word_count`) decorated with `@tool`.
- **`agent/graph.py`** — builds the ReAct agent with `create_react_agent`,
  a `ChatOllama` model, the tools, and the checkpointer. Exposes `run()`
  (blocking) and `stream_tokens()` (yields the model's `AIMessageChunk`
  tokens only, so raw tool output never leaks into the reply).
- **`agent/checkpointer.py`** — a process-wide `PostgresSaver` backed by a
  connection pool; `thread_id` is the conversation key.
- **`chat/models.py`** — `Thread` / `Message` mirror the conversation into the
  ORM for the admin and UI. The `thread_id` matches the checkpointer key.
- **`chat/views.py`** — `create_thread`, `chat` (blocking JSON),
  `stream` (SSE), and `thread_messages`.
- **`templates/chat/index.html`** — a dependency-free chat page using `fetch`
  and `EventSource`.

## Endpoints

| Method | Path                                       | Purpose                     |
|--------|--------------------------------------------|-----------------------------|
| GET    | `/`                                        | Chat UI                     |
| POST   | `/api/threads/`                            | Create a thread             |
| GET    | `/api/threads/<uuid>/messages/`            | List a thread's messages    |
| POST   | `/api/chat/`                               | Blocking turn (JSON reply)  |
| GET    | `/api/threads/<uuid>/stream/?message=...`  | Streaming turn (SSE)        |

## Tests

```bash
uv run python manage.py test              # fast, offline (agent calls are mocked)
RUN_AGENT_TESTS=1 uv run python manage.py test chat.tests.AgentSmokeTest   # live Ollama
```

## Code quality

Linting/formatting is [ruff](https://docs.astral.sh/ruff/) and type checking is
[mypy](https://mypy-lang.org/); both are configured in `pyproject.toml`.

```bash
uv run ruff check .           # lint (PEP 8 / pyflakes / import order / pyupgrade)
uv run ruff format .          # auto-format (add --check to only verify)
uv run mypy .                 # static type check
```

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request to `main` (and on pushes
to `main`). It spins up a Postgres service, installs deps with `uv sync
--frozen`, then runs, in order: `ruff check`, `ruff format --check`, `mypy`,
`manage.py check`, and the test suite. The live Ollama test is skipped in CI
(no model available) — it only runs locally with `RUN_AGENT_TESTS=1`.

## Configuration

All via environment variables (see `.env.example`): Django secret/debug/hosts,
Postgres connection, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and an optional
`CHECKPOINTER_DSN` (defaults to the same Postgres database).

## Notes & ideas to extend

- Swap `llama3.2` for `qwen2.5` (stronger tool use) by setting `OLLAMA_MODEL`.
- Add real tools (web search, DB lookups over your Django models).
- Serve under ASGI (`uvicorn config.asgi:application`) for true async streaming.
- Add per-user threads by attaching `Thread` to `request.user`.

<img width="756" height="643" alt="image" src="https://github.com/user-attachments/assets/e44e60aa-4400-4522-9be7-1bc6723ed377" />

