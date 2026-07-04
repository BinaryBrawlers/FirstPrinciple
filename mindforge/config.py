"""Application configuration via pydantic-settings.

All settings are read from environment variables or a .env file.
Uses Mistral AI for both LLM (generation) and embeddings via LiteLLM routing.
Cognee runs entirely locally (SQLite + LanceDB + Ladybug/Kuzu).

LLM_MODEL format: "mistral/mistral-small-latest"
  - The "mistral/" prefix is required by LiteLLM/Cognee for routing.
  - MindForge agents strip this prefix when calling the Mistral SDK directly
    (e.g. settings.llm_model_name → "mistral-small-latest").
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

    # ── Mistral API key ────────────────────────────────────────────────────────
    # Used directly by MindForge agents via the mistralai SDK.
    # Also echoed as LLM_API_KEY and EMBEDDING_API_KEY for Cognee/LiteLLM.
    mistral_api_key: str = ""

    # ── Cognee — LLM (Mistral via LiteLLM / "custom" provider) ────────────────
    # LLM_MODEL must include the "mistral/" prefix for LiteLLM routing.
    # Agents use .llm_model_name (prefix stripped) when calling the Mistral SDK.
    llm_provider: str = "custom"
    llm_model: str = "mistral/mistral-small-latest"
    llm_api_key: str = ""

    # ── Cognee — Embeddings (Mistral via LiteLLM / "custom" provider) ─────────
    embedding_provider: str = "custom"
    embedding_model: str = "mistral/mistral-embed"
    embedding_dimensions: int = 1024
    embedding_api_key: str = ""

    # ── Cognee — Local backend configuration ──────────────────────────────────
    db_provider: str = "sqlite"
    vector_db_provider: str = "lancedb"
    graph_database_provider: str = "ladybug"

    # ── Cognee — Storage paths ─────────────────────────────────────────────────
    system_root_directory: str = ""
    data_root_directory: str = ""

    # ── Cognee — Misc ──────────────────────────────────────────────────────────
    cognee_skip_connection_test: bool = True
    litellm_drop_params: bool = True
    caching: bool = True
    cache_backend: str = "fs"
    telemetry_disabled: bool = True

    # ── FastAPI server ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def llm_model_name(self) -> str:
        """Return the model name without the LiteLLM provider prefix.

        Used when calling the Mistral SDK directly (it does not accept the
        "mistral/" prefix that LiteLLM/Cognee expects).

        Example:
            "mistral/mistral-small-latest" → "mistral-small-latest"
        """
        return self.llm_model.removeprefix("mistral/")

    @property
    def effective_mistral_api_key(self) -> str:
        """Return the best available Mistral API key.

        Prefers MISTRAL_API_KEY, falls back to LLM_API_KEY.
        """
        return self.mistral_api_key or self.llm_api_key


# Module-level singleton — import and use `settings` directly.
settings = Settings()
