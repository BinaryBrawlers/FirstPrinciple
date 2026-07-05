"""
config.py — Environment configuration and cognee initialisation.

IMPORTANT: This module sets COGNEE_SKIP_CONNECTION_TEST before importing cognee,
which is required to allow startup without an active cognee instance.
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# --- Required environment variables ---
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY is not set. "
        "Copy .env.example to .env and fill in your key."
    )

# Must be set BEFORE importing cognee so the connection test is skipped at import time.
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

# Now it is safe to import cognee.
import cognee  # noqa: E402

# Configure cognee to use Mistral as the LLM provider and fastembed for embeddings.
cognee.config.set_llm_provider("mistral")
cognee.config.set_embedding_provider("fastembed")
