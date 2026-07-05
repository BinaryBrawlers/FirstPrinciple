"""
Application configuration — must be imported before any cognee import.

Sets required environment variables and initialises the cognee LLM/embedding
providers so the rest of the application can import cognee safely.
"""
import os
from dotenv import load_dotenv

# Load .env file if present (development convenience)
load_dotenv()

# Must be set before cognee is imported anywhere in the process
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

# Validate required env vars
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    import warnings
    warnings.warn(
        "MISTRAL_API_KEY is not set. cognee LLM calls will fail at runtime.",
        RuntimeWarning,
        stacklevel=1,
    )

# Now safe to import and configure cognee
import cognee  # noqa: E402

cognee.config.set_llm_provider("mistral")
cognee.config.set_embedding_provider("fastembed")
