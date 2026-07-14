# syntax=docker/dockerfile:1

# ---- builder: install deps + project into a self-contained .venv ----------
# We build against the same python image used at runtime so the resulting
# .venv is portable (its interpreter path exists in the runtime stage).
FROM python:3.14-slim-bookworm AS builder

# Bring in the uv binary from Astral's published image.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first, using cache + bind mounts so this layer is
# reused unless pyproject.toml / uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime: minimal image with just the venv + code ---------------------
FROM python:3.14-slim-bookworm AS runtime

# Run as a non-root user.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# Ensure the app user owns the tree (incl. the /app dir node itself) and has a
# writable STATIC_ROOT for collectstatic.
RUN mkdir -p /app/staticfiles && chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=config.settings

USER app
EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "config.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
