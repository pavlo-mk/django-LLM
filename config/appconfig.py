"""Typed application configuration.

All environment-driven config lives here as a validated pydantic-settings
model instead of scattered ``os.environ.get`` calls. Field names map to
env vars case-insensitively (``ollama_model`` <- ``OLLAMA_MODEL``). Values are
read from the process environment first, then a local ``.env`` file.
"""

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Django
    django_secret_key: str = "django-insecure-change-me"
    django_debug: bool = True
    django_allowed_hosts: str = "localhost,127.0.0.1"

    # Postgres
    postgres_db: str = "djllm"
    postgres_user: str = "djllm"
    postgres_password: str = "djllm"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    # Ollama / agent
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: float = 120.0  # seconds per model request

    # Observability (all optional)
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1
    # LangSmith tracing is enabled purely via env (LANGSMITH_TRACING=true,
    # LANGSMITH_API_KEY=...); LangChain reads those itself, no code needed.

    # Overrides the derived DSN below if set (e.g. a managed connection string).
    checkpointer_dsn: str = ""

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.django_allowed_hosts.split(",") if h.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_dsn(self) -> str:
        if self.checkpointer_dsn:
            return self.checkpointer_dsn
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = AppSettings()
