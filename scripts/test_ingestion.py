"""
Ingestion Agent smoke test — exercises decompose_topic and fetch_wikipedia
against the live Mistral API and Wikipedia.

Run from the project root:
    source backend/.env/bin/activate
    python scripts/test_ingestion.py

Or pass a custom topic:
    python scripts/test_ingestion.py "deep learning"
"""
import asyncio
import sys
import os

# Put backend/ on the path so bare imports (config, agents.*, etc.) resolve
_backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(_backend_dir))

# config MUST be imported first — loads .env and patches litellm
import config  # noqa: F401, E402

from agents.ingestion import decompose_topic, fetch_wikipedia


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char="─", width=60):
    print(char * width)


def _header(title: str):
    _hr()
    print(f"  {title}")
    _hr()


# ---------------------------------------------------------------------------
# Test: decompose_topic
# ---------------------------------------------------------------------------

async def test_decompose_topic(topic: str):
    _header(f"decompose_topic({topic!r})")
    subtopics = await decompose_topic(topic)

    print(f"  Returned {len(subtopics)} subtopic(s):")
    for i, s in enumerate(subtopics, 1):
        print(f"    {i:2}. {s}")

    assert isinstance(subtopics, list), "Expected a list"
    assert len(subtopics) >= 1, "Expected at least one subtopic"
    assert all(isinstance(s, str) and s.strip() for s in subtopics), \
        "All subtopics must be non-empty strings"
    print("\n  ✅ decompose_topic passed")
    return subtopics


# ---------------------------------------------------------------------------
# Test: fetch_wikipedia
# ---------------------------------------------------------------------------

async def test_fetch_wikipedia(subtopic: str):
    _header(f"fetch_wikipedia({subtopic!r})")
    episodes = await fetch_wikipedia(subtopic)

    print(f"  Returned {len(episodes)} episode(s):")
    for ep in episodes:
        print(f"\n    id              : {ep.id}")
        print(f"    concept         : {ep.concept}")
        print(f"    outcome         : {ep.outcome.value}")
        print(f"    source_confidence: {ep.source_confidence.value}")
        print(f"    source          : {ep.source}")
        print(f"    published_date  : {ep.published_date}")
        print(f"    requires        : {ep.requires}")
        print(f"    concurrent_with : {ep.concurrent_with}")
        print(f"    problem_posed   : {ep.problem_posed[:120]}...")

    # Structural assertions
    for ep in episodes:
        assert ep.id, "Episode must have a non-empty id"
        assert ep.concept, "Episode must have a non-empty concept"
        assert ep.problem_posed, "Episode must have a problem_posed"
        assert ep.attempted_solution, "Episode must have an attempted_solution"
        assert ep.why, "Episode must have a why"
        from models.schemas import SourceConfidence
        assert ep.source_confidence == SourceConfidence.NAMED_REFERENCE, \
            f"Wikipedia episodes must have source_confidence=named_reference, got {ep.source_confidence}"

    print("\n  ✅ fetch_wikipedia passed")
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(topic: str):
    print(f"\n🧪 Ingestion smoke test — topic: {topic!r}\n")

    # Step 1: decompose
    subtopics = await test_decompose_topic(topic)

    # Step 2: fetch Wikipedia for the first subtopic
    first = subtopics[0]
    print(f"\n  (testing fetch_wikipedia on first subtopic: {first!r})\n")
    episodes = await test_fetch_wikipedia(first)

    # Summary
    _hr("═")
    print(f"  🎉 All checks passed!")
    print(f"     topic     : {topic}")
    print(f"     subtopics : {len(subtopics)}")
    print(f"     episodes  : {len(episodes)} (from {first!r})")
    _hr("═")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "operating system memory management"
    asyncio.run(main(topic))
