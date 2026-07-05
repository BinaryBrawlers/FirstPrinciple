"""Property-based test for exponential backoff timing in fetch_with_retry.

Property covered:
  Property 9 — Exponential backoff timing   (Validates: Requirements 4.8)

The test verifies:
  - For attempt k ∈ {0, 1, 2} the sleep duration equals exactly 2^k seconds.
  - The ``tenacity`` library is NOT imported anywhere in agents/ingestion.py.
"""
from __future__ import annotations

import asyncio
import ast
import importlib.util
import os
import sys
from typing import List
from unittest.mock import AsyncMock, patch, call

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the backend package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.ingestion import TransientFetchError, fetch_with_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def _ingestion_module_source() -> str:
    """Return the raw source of agents/ingestion.py."""
    module_path = os.path.join(os.path.dirname(__file__), "..", "agents", "ingestion.py")
    with open(os.path.normpath(module_path)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Property 9: Exponential backoff timing
#
# For attempt k ∈ {0, 1, 2}, the sleep duration passed to asyncio.sleep()
# SHALL equal 2^k seconds.  tenacity SHALL NOT be imported.
#
# **Validates: Requirements 4.8**
# ---------------------------------------------------------------------------

@given(num_failures=st.integers(min_value=1, max_value=3))
@settings(max_examples=50)
def test_backoff_sleep_durations_are_powers_of_two(num_failures: int):
    """Sleep between consecutive failures equals 2^attempt seconds.

    num_failures controls how many TransientFetchError raises occur before
    either succeeding (num_failures < 3) or exhausting all attempts
    (num_failures == 3).  In every case the recorded sleep calls must equal
    [2^0, 2^1, ...] for each inter-attempt gap.
    """
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= num_failures:
            raise TransientFetchError("simulated failure")
        return "ok"

    sleep_calls: List[float] = []

    async def fake_sleep(seconds: float):
        sleep_calls.append(seconds)

    with patch("agents.ingestion.asyncio.sleep", side_effect=fake_sleep):
        if num_failures < 3:
            result = run(fetch_with_retry(flaky, max_attempts=3))
            assert result == "ok"
        else:
            with pytest.raises(TransientFetchError):
                run(fetch_with_retry(flaky, max_attempts=3))

    # Number of sleeps = number of failures that had a subsequent retry attempt
    expected_sleep_count = min(num_failures, 2)  # max 2 sleeps for 3 attempts
    assert len(sleep_calls) == expected_sleep_count, (
        f"Expected {expected_sleep_count} sleep call(s), got {len(sleep_calls)}"
    )

    for k, duration in enumerate(sleep_calls):
        expected = float(2 ** k)
        assert duration == expected, (
            f"Attempt {k}: expected sleep of {expected}s, got {duration}s"
        )


def test_tenacity_not_imported_in_ingestion_module():
    """agents/ingestion.py SHALL NOT import tenacity (Requirements 4.8)."""
    source = _ingestion_module_source()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "tenacity" and not alias.name.startswith("tenacity."), (
                    "agents/ingestion.py imports 'tenacity', which is forbidden by Requirements 4.8"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "tenacity" and (
                node.module is None or not node.module.startswith("tenacity.")
            ), (
                "agents/ingestion.py imports from 'tenacity', which is forbidden by Requirements 4.8"
            )


@given(attempt=st.integers(min_value=0, max_value=1))
@settings(max_examples=20)
def test_single_attempt_sleep_equals_2_pow_attempt(attempt: int):
    """For inter-attempt gap k → k+1, the sleep duration is exactly 2^k seconds.

    With max_attempts=3 there are exactly two sleeps (after attempt 0 and after
    attempt 1).  Attempt 2 is the last so no sleep follows it.
    """
    async def always_fail():
        raise TransientFetchError("always fail")

    sleep_calls: List[float] = []

    async def fake_sleep(seconds: float):
        sleep_calls.append(seconds)

    with patch("agents.ingestion.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(TransientFetchError):
            run(fetch_with_retry(always_fail, max_attempts=3))

    # Exactly two sleeps for three attempts
    assert len(sleep_calls) == 2
    assert sleep_calls[attempt] == float(2 ** attempt), (
        f"Sleep after attempt {attempt} should be {2 ** attempt}s, "
        f"got {sleep_calls[attempt]}s"
    )
