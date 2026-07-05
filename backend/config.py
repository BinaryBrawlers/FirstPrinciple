"""
Application configuration — must be imported before any cognee import.

Loads .env, applies the LiteLLM param-strip patch (removes Cognee-internal
kwargs that Mistral's API rejects), then lets cognee auto-configure itself
from the environment variables set in .env.
"""
import os
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Must be set before cognee is imported anywhere in the process
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

# ---------------------------------------------------------------------------
# LiteLLM patch — strip Cognee-internal params that Mistral's API rejects
# ---------------------------------------------------------------------------
import litellm  # noqa: E402

# Drop any unsupported params globally (e.g. dimensions, dataset_name) rather
# than letting them surface as UnsupportedParamsError at call time.
litellm.drop_params = True

_orig_acompletion = litellm.acompletion


async def _clean_acompletion(*args, **kwargs):
    kwargs.pop("dataset_name", None)
    kwargs.pop("dataset_id", None)
    return await _orig_acompletion(*args, **kwargs)


litellm.acompletion = _clean_acompletion

_orig_completion = litellm.completion


def _clean_completion(*args, **kwargs):
    kwargs.pop("dataset_name", None)
    kwargs.pop("dataset_id", None)
    return _orig_completion(*args, **kwargs)


litellm.completion = _clean_completion

# ---------------------------------------------------------------------------
# Validate key presence (warn only — don't crash on import)
# ---------------------------------------------------------------------------
_api_key = os.environ.get("LLM_API_KEY") or os.environ.get("MISTRAL_API_KEY", "")
if not _api_key:
    import warnings
    warnings.warn(
        "LLM_API_KEY / MISTRAL_API_KEY is not set. cognee LLM calls will fail at runtime.",
        RuntimeWarning,
        stacklevel=1,
    )

# cognee reads LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, EMBEDDING_PROVIDER,
# EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_DIMENSIONS etc. directly
# from the environment — no explicit set_llm_provider() call needed when
# using the "custom" (LiteLLM) provider.
