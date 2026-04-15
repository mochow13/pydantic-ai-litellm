#!/usr/bin/env python3
"""
Real-world style example using the package as published on PyPI.

Install
-------
    pip install pydantic-ai-litellm

Environment
-----------
    export OPENAI_API_KEY="sk-..."           # required for the default model
    export MODEL_NAME="gpt-4o-mini"          # optional; any LiteLLM-supported id

Run
---
    python examples/07_install_from_pypi.py

You can point MODEL_NAME at other providers (e.g. ``anthropic/claude-3-5-haiku-20241022``)
if you set the API key env vars those providers expect; see LiteLLM provider docs.

From a git checkout (without ``pip install -e .``), run from the repository root::

    python examples/07_install_from_pypi.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Running this file from a clone? Put the repo root on sys.path when the package
# is not installed (``pip install pydantic-ai-litellm`` makes this a no-op in practice).
_root = Path(__file__).resolve().parent.parent
if (_root / "pydantic_ai_litellm").is_dir():
    sys.path.insert(0, str(_root))

from pydantic_ai import Agent

from pydantic_ai_litellm import LiteLLMModel


def lookup_order_status(order_id: str) -> str:
    """Return a fake fulfillment status for a retail order id (demo only)."""
    if not order_id.strip():
        return "No order id provided."
    # Deterministic mock so the behavior is easy to reason about.
    suffix = order_id.strip()[-1].lower()
    if suffix in "aeiou":
        return f"Order {order_id}: shipped"
    return f"Order {order_id}: processing"


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Missing OPENAI_API_KEY. Export it, then run again.\n"
            "Example: export OPENAI_API_KEY=$(cat ~/.openai_key)",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model = LiteLLMModel(model_name=model_name, api_key=api_key)
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a concise support assistant. "
            "Use tools when they help answer factual or status questions."
        ),
        tools=[lookup_order_status],
    )

    result = await agent.run(
        "What's going on with order ORD-1001A? One short sentence with the status."
    )
    print(result.output)
    print(f"usage: {result.usage()}")


if __name__ == "__main__":
    asyncio.run(main())
