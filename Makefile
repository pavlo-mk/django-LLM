.PHONY: install run test lint format typecheck check up down logs docker-build

install:            ## Install deps + dev tools and git hooks
	uv sync --dev
	uv run pre-commit install

run:                ## Run the ASGI server locally (async streaming works here)
	uv run uvicorn config.asgi:application --reload

test:               ## Run the test suite
	uv run pytest

lint:               ## Ruff lint
	uv run ruff check .

format:             ## Ruff auto-format
	uv run ruff format .

typecheck:          ## mypy
	uv run mypy .

check: lint typecheck test   ## Everything CI runs

up:                 ## Start the full stack (db + app) in Docker
	docker compose up -d --build

down:               ## Stop the stack
	docker compose down

logs:               ## Tail app logs
	docker compose logs -f app

docker-build:       ## Build the app image
	docker build -t djllm-app .
