#!/usr/bin/env python3
"""
Quick Start Example - Basic usage of pydantic-ai-litellm
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

async def main():
    """Quick start example showing basic usage."""
    
    # Initialize with any LiteLLM-supported model
    model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Create an agent
    agent = Agent(model=model)

    # Run inference
    result = await agent.run("What is the capital of France?")
    print(f"Result: {result.output}")
    print(f"Usage: {result.usage()}")

if __name__ == "__main__":
    asyncio.run(main())
