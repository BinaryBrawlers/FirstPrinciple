"""
Cognee smoke test — verifies Cognee works end-to-end with Mistral + local backends.

Run from the project root:
    python scripts/cognee_smoke_test.py
"""
import asyncio
import sys
import os

# Ensure project root is on the path so backend.config loads .env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# config.py must be imported first — it loads .env and patches litellm
import backend.config  # noqa: F401, E402

import cognee
from cognee.api.v1.search import SearchType


async def main():
    # 1. Prune any previous state (safe on first run)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("✅ Step 1/5: Pruned old data")

    # 2. Add text
    text = (
        "The Perceptron was invented by Frank Rosenblatt in 1958. "
        "It is the simplest form of a neural network, consisting of "
        "a single layer of weights and a threshold activation function. "
        "Backpropagation was published by Rumelhart, Hinton, and Williams in 1986."
    )
    await cognee.add(text, dataset_name="test_deep_learning")
    print("✅ Step 2/5: Added text to dataset")

    # 3. Cognify (build knowledge graph + embeddings)
    await cognee.cognify(dataset_name="test_deep_learning")
    print("✅ Step 3/5: Cognified (graph + embeddings built)")

    # 4. Search (recall)
    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Who invented the Perceptron?",
        datasets=["test_deep_learning"],
    )
    print("✅ Step 4/5: Search returned results:")
    for r in results:
        print(f"   → {r}")

    # 5. Clean up
    await cognee.prune.prune_data()
    print("✅ Step 5/5: Cleaned up test data")

    print("\n🎉 All steps passed — Cognee + Mistral is working!")


if __name__ == "__main__":
    asyncio.run(main())
