#!/usr/bin/env python3
"""
Custom API Endpoints Example - Using custom endpoints with pydantic-ai-litellm
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

async def main():
    """Example showing custom API endpoints configuration."""
    
    # Example 1: Custom OpenAI-compatible endpoint
    model = LiteLLMModel(
        model_name=os.getenv("ANTHROPIC_MODEL"),
        api_base=os.getenv("LITELLM_API_BASE"),
        api_key=os.getenv("LITELLM_API_KEY"),
        custom_llm_provider="",
    )

    agent = Agent(model=model)
    
    try:
        result = await agent.run("Hello! Can you tell me about yourself?")
        print(f"Custom endpoint result: {result.output}")
        print(f"Usage: {result.usage()}")
    except Exception as e:
        print(f"Error with custom endpoint: {e}")
        print("Note: This example requires a valid custom endpoint URL and API key")
    
if __name__ == "__main__":
    asyncio.run(main())
