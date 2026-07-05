"""
FastAPI application entry point.

- Registers the chat and ingest routers.
- On startup, calls seed_tracks_if_absent() to load hand-authored seed episodes
  into Track A if they are not already present (Requirement 10.3).

Requirements: 10.3, 11.1, 11.2
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# config MUST be imported before any cognee import
import config  # noqa: F401

from fastapi import FastAPI

from memory.seed import seed_tracks_if_absent
from routers.chat import router as chat_router
from routers.ingest import router as ingest_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan hook — runs seed loader on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup:
      - Calls seed_tracks_if_absent() to ensure Track A contains the OS memory
        management and deep learning seed episodes before the first request is
        served.  The call is guarded internally against re-seeding if episodes
        are already present (idempotent).

    Shutdown:
      - Nothing to tear down for the demo scope.

    Requirements: 10.3
    """
    logger.info("startup: seeding Track A with hand-authored episodes if absent …")
    try:
        await seed_tracks_if_absent()
        logger.info("startup: seed check complete.")
    except Exception as exc:  # noqa: BLE001
        # Log and continue — the app should still start even if cognee is
        # temporarily unreachable.  Chat requests will still work against
        # whatever is already in Track A.
        logger.error(
            "startup: seed_tracks_if_absent() raised an exception: %s — "
            "continuing without guaranteed seed data.",
            exc,
        )

    yield  # application is now running

    # --------------- shutdown ---------------
    logger.info("shutdown: cleanup complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FirstPrinciple",
    description=(
        "Multi-agent learning system that guides learners through the historical "
        "development of technical concepts via Socratic dialogue and adversarial "
        "testing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

# POST /chat — SSE token stream (Requirement 11.1)
app.include_router(chat_router)

# POST /ingest, POST /session/start, POST /session/end (Requirement 11.2)
app.include_router(ingest_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Lightweight liveness probe for Docker / load-balancer health checks."""
    return {"status": "ok"}
