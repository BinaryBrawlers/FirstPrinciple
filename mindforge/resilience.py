"""Resilient wrappers around Cognee's four core API calls.

Each wrapper retries up to 3 times with exponential backoff using tenacity.
On final failure:
  - safe_recall   → returns [] (never re-raises)
  - safe_remember → writes to LocalFallbackCache, then re-raises
  - safe_improve  → logs error and re-raises
  - safe_forget   → logs error and re-raises

Requirements: 18.1, 18.2, 18.3
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Optional cognee import — may not be installed in dev environments.
# All wrappers handle ImportError at call-time via the _get_cognee() helper.
# ---------------------------------------------------------------------------
try:
    import cognee as _cognee  # noqa: F401
except ImportError:  # pragma: no cover
    _cognee = None  # type: ignore[assignment]

logger = logging.getLogger("mindforge.resilience")

# ---------------------------------------------------------------------------
# Shared retry configuration
# ---------------------------------------------------------------------------
_STOP = stop_after_attempt(3)
_WAIT = wait_exponential(multiplier=1, min=1, max=10)


def _get_cognee() -> Any:
    """Return the cognee module, raising ImportError with a helpful message if missing."""
    if _cognee is None:
        raise ImportError(
            "cognee is not installed. Run: pip install cognee"
        )
    return _cognee


def _retry_context() -> AsyncRetrying:
    """Build a fresh AsyncRetrying context with the standard policy."""
    return AsyncRetrying(
        stop=_STOP,
        wait=_WAIT,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# safe_remember
# ---------------------------------------------------------------------------
async def safe_remember(data: Any, **kwargs: Any) -> Any:
    """Call cognee.remember() with retry; on failure write to LocalFallbackCache.

    Args:
        data:    The content/data to store in Cognee.
        **kwargs: Forwarded directly to cognee.remember()
                  (dataset, session_id, self_improvement, …).

    Returns:
        Whatever cognee.remember() returns.

    Raises:
        The final exception after all retries are exhausted, after queuing
        the failed write to LocalFallbackCache.
    """
    cognee = _get_cognee()
    # Installed Cognee uses `dataset_name` not `dataset` as the kwarg name.
    if "dataset" in kwargs and "dataset_name" not in kwargs:
        kwargs["dataset_name"] = kwargs.pop("dataset")
    try:
        async for attempt in _retry_context():
            with attempt:
                return await cognee.remember(data, **kwargs)
    except Exception as exc:
        logger.error(
            "safe_remember failed after all retries. "
            "Queuing write to LocalFallbackCache. Error: %s",
            exc,
        )
        # Lazy import to avoid circular dependency with mindforge.cache (Task 4.2).
        # If cache.py doesn't exist yet this block is silently skipped so that
        # resilience.py can be imported independently.
        try:
            from mindforge.cache import LocalFallbackCache  # noqa: PLC0415

            cache = LocalFallbackCache()
            await cache.store(data, kwargs)
        except Exception as cache_exc:
            logger.warning(
                "LocalFallbackCache unavailable; skipping fallback store. Error: %s",
                cache_exc,
            )
        raise


# ---------------------------------------------------------------------------
# safe_recall
# ---------------------------------------------------------------------------
async def safe_recall(query_text: str, **kwargs: Any) -> list:
    """Call cognee.recall() with retry; on failure return [] instead of raising.

    Args:
        query_text: The natural-language query to pass to Cognee.
        **kwargs:   Forwarded directly to cognee.recall()
                    (dataset, session_id, limit, …).

    Returns:
        List of recall results, or [] if all retries fail.
    """
    cognee = _get_cognee()
    try:
        async for attempt in _retry_context():
            with attempt:
                return await cognee.recall(query_text, **kwargs)
    except Exception as exc:
        logger.error(
            "safe_recall failed after all retries; returning []. Error: %s",
            exc,
        )
        return []
    # Unreachable, but satisfies type checkers.
    return []  # pragma: no cover


# ---------------------------------------------------------------------------
# safe_improve
# ---------------------------------------------------------------------------
async def safe_improve(
    dataset: str | None = None,
    session_ids: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call cognee.improve() with retry; on failure log and re-raise.

    Args:
        dataset:     Dataset name to improve, or None for the default dataset.
        session_ids: List of session IDs whose memories should be consolidated.
        **kwargs:    Additional keyword arguments forwarded to cognee.improve().

    Returns:
        Whatever cognee.improve() returns.

    Raises:
        The final exception after all retries are exhausted.
    """
    cognee = _get_cognee()
    try:
        async for attempt in _retry_context():
            with attempt:
                return await cognee.improve(
                    dataset=dataset, session_ids=session_ids, **kwargs
                )
    except Exception as exc:
        logger.error(
            "safe_improve failed after all retries. Error: %s",
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# safe_forget
# ---------------------------------------------------------------------------
async def safe_forget(
    data_item: Any = None,
    dataset: str | None = None,
    everything: bool = False,
    **kwargs: Any,
) -> Any:
    """Call cognee.forget() with retry; on failure log and re-raise.

    Args:
        data_item:  Specific data item / identifier to remove, or None.
        dataset:    Dataset to delete, or None.
        everything: If True, wipe all stored memories.
        **kwargs:   Additional keyword arguments forwarded to cognee.forget().

    Returns:
        Whatever cognee.forget() returns.

    Raises:
        The final exception after all retries are exhausted.
    """
    cognee = _get_cognee()
    try:
        async for attempt in _retry_context():
            with attempt:
                return await cognee.forget(
                    data_item=data_item,
                    dataset=dataset,
                    everything=everything,
                    **kwargs,
                )
    except Exception as exc:
        logger.error(
            "safe_forget failed after all retries. Error: %s",
            exc,
        )
        raise
