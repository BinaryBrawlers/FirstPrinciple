"""Local fallback cache for failed Cognee writes.

When ``safe_remember`` cannot reach Cognee after all retries it calls
``LocalFallbackCache.store()`` to persist the payload in a local SQLite
database.  At every session start ``flush()`` is called to replay any queued
writes against Cognee and clean up successfully replayed rows.

Schema
------
::

    pending_writes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        data       TEXT     NOT NULL,  -- JSON-serialised payload
        kwargs     TEXT     NOT NULL,  -- JSON-serialised cognee.remember() kwargs
        created_at TEXT     NOT NULL   -- ISO-8601 UTC timestamp
    )

Requirements: 18.4
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger("mindforge.cache")

# Default database path — can be overridden via the MINDFORGE_CACHE_DB env var.
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".mindforge_fallback_cache.db"
)


class LocalFallbackCache:
    """SQLite-backed queue for Cognee writes that failed at runtime.

    Usage::

        cache = LocalFallbackCache()          # uses default DB path
        await cache.store(data, kwargs)       # queue a failed write
        await cache.flush()                   # replay queued writes (call at session start)

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to the value of the
        ``MINDFORGE_CACHE_DB`` environment variable, falling back to
        ``../.mindforge_fallback_cache.db`` relative to this module's directory.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path: str = (
            db_path
            or os.environ.get("MINDFORGE_CACHE_DB", "")
            or os.path.abspath(_DEFAULT_DB_PATH)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_conn(self) -> aiosqlite.Connection:
        """Open (or reopen) a connection and ensure the schema exists."""
        conn = await aiosqlite.connect(self._db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_writes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                data       TEXT    NOT NULL,
                kwargs     TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )
        await conn.commit()
        return conn

    @staticmethod
    def _serialise(obj: Any) -> str:
        """JSON-serialise *obj*, falling back to its string representation."""
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(obj))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(self, data: Any, cognee_kwargs: dict[str, Any]) -> None:
        """Queue a failed Cognee write for later replay.

        Parameters
        ----------
        data:
            The payload that was passed to ``cognee.remember()``.
        cognee_kwargs:
            The keyword arguments that were forwarded to ``cognee.remember()``
            (e.g. ``dataset``, ``session_id``, ``self_improvement``).
        """
        now = datetime.now(timezone.utc).isoformat()
        serialised_data = self._serialise(data)
        serialised_kwargs = self._serialise(cognee_kwargs)

        conn = await self._get_conn()
        try:
            await conn.execute(
                "INSERT INTO pending_writes (data, kwargs, created_at) VALUES (?, ?, ?)",
                (serialised_data, serialised_kwargs, now),
            )
            await conn.commit()
            logger.info(
                "LocalFallbackCache: queued write (created_at=%s, data_preview=%.80s)",
                now,
                serialised_data,
            )
        except Exception as exc:
            logger.error("LocalFallbackCache.store failed: %s", exc)
            raise
        finally:
            await conn.close()

    async def flush(self) -> int:
        """Replay all queued writes against Cognee; delete rows that succeed.

        This method is intended to be called at every session start so that
        writes that failed during a previous session are retried before new
        interactions begin.

        Returns
        -------
        int
            Number of rows successfully replayed and deleted.
        """
        # Lazy import — cognee may not be present in test environments.
        try:
            import cognee  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "LocalFallbackCache.flush: cognee is not installed; skipping flush."
            )
            return 0

        conn = await self._get_conn()
        try:
            async with conn.execute(
                "SELECT id, data, kwargs FROM pending_writes ORDER BY id"
            ) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.error("LocalFallbackCache.flush: failed to read rows: %s", exc)
            await conn.close()
            return 0

        if not rows:
            await conn.close()
            return 0

        logger.info(
            "LocalFallbackCache.flush: replaying %d pending write(s).", len(rows)
        )

        replayed = 0
        for row in rows:
            row_id: int = row["id"]
            try:
                data = json.loads(row["data"])
                kwargs: dict[str, Any] = json.loads(row["kwargs"])
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "LocalFallbackCache.flush: row %d has invalid JSON, skipping: %s",
                    row_id,
                    exc,
                )
                continue

            try:
                await cognee.remember(data, **kwargs)
                # Delete the row only on success.
                await conn.execute(
                    "DELETE FROM pending_writes WHERE id = ?", (row_id,)
                )
                await conn.commit()
                replayed += 1
                logger.info(
                    "LocalFallbackCache.flush: row %d replayed and deleted.", row_id
                )
            except Exception as exc:
                # Leave the row in the queue; it will be retried next session.
                logger.warning(
                    "LocalFallbackCache.flush: row %d replay failed, keeping: %s",
                    row_id,
                    exc,
                )

        await conn.close()
        logger.info(
            "LocalFallbackCache.flush: finished — %d/%d rows replayed.", replayed, len(rows)
        )
        return replayed
