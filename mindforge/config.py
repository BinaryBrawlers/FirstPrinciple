"""Application configuration via pydantic-settings.

All settings are read from environment variables or a .env file.
Uses Mistral API for both LLM (generation) and embeddings.
Cognee runs locally — no Cognee cloud account required.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MindForge runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Mistral API ────────────────────────────────────────────────────────────
    # Single key used for both LLM and embeddings.
    # Get yours at: https://console.mistral.ai/api-keys
    mistral_api_key: str = ""

    # ── Cognee — LLM (Mistral via LiteLLM) ────────────────────────────────────
    # Cognee routes LLM calls through LiteLLM; the "mistral/" prefix is required.
    llm_provider: str = "litellm"
    llm_model: str = "mistral/mistral-small-latest"
    llm_api_key: str = ""  # set to MISTRAL_API_KEY value in .env

    # ── Cognee — Embeddings (Mistral) ─────────────────────────────────────────
    # mistral-embed is Mistral's dedicated embedding model (1024 dimensions).
    # IMPORTANT: both LLM and embedding must be configured; if only one is set,
    # the other silently falls back to OpenAI.
    embedding_provider: str = "mistral"
    embedding_model: str = "mistral-embed"
    embedding_dimensions: int = 1024

    # ── FastAPI server ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]


# Module-level singleton — import and use `settings` directly.
settings = Settings()
