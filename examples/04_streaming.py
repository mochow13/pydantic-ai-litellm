#!/usr/bin/env python3
"""
Streaming Example - Real-time text streaming with pydantic-ai-litellm
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

async def main():
    """Example showing streaming functionality."""
    
    model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    agent = Agent(model=model)

    try:
        print("Streaming response:")
        print("-" * 50)
        
        # Stream a response with delta=True to get incremental updates
        async with agent.run_stream("Write a poem about AI") as stream:
            async for text in stream.stream_text(delta=True):
                print(text, end="", flush=True)
        
        print("\n" + "-" * 50)
        print("Streaming completed.")
        
        # Get the final result
        print(f"\nStreaming usage available in the stream context")
        
    except Exception as e:
        print(f"Streaming failed: {e}")
        print("Make sure you have a valid API key for a model that supports streaming")

if __name__ == "__main__":
    asyncio.run(main())
