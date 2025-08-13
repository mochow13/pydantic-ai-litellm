#!/usr/bin/env python3
"""
Configuration Example - Using model settings with pydantic-ai-litellm
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel, LiteLLMModelSettings

async def main():
    """Example showing different configuration options."""
    
    # Example 1: Using settings dictionary
    print("=== Configuration with Settings Dictionary ===")
    
    settings: LiteLLMModelSettings = {
        'temperature': 0.7,
        'max_tokens': 1000,
        'litellm_api_key': os.getenv('GEMINI_API_KEY'),
        'litellm_api_base': 'https://generativelanguage.googleapis.com',
        'extra_headers': {'Custom-Header': 'value'}
    }

    model = LiteLLMModel("gemini/gemini-2.5-flash", settings=settings)
    agent = Agent(model=model)
    
    try:
        result = await agent.run("Write a short creative story in exactly 100 words")
        print(f"Result: {result.output}")
        print(f"Usage: {result.usage()}")
        print()
        
    except Exception as e:
        print(f"Configuration example 1 failed: {e}")
        print()
    
    # Example 2: Temperature comparison
    print("=== Temperature Comparison ===")
    
    # Low temperature (more deterministic)
    low_temp_model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        settings={'temperature': 0.1}
    )
    
    # High temperature (more creative)
    high_temp_model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        settings={'temperature': 0.9}
    )
    
    prompt = "Complete this sentence: The future of AI is"
    
    try:
        # Low temperature response
        low_temp_agent = Agent(model=low_temp_model)
        result1 = await low_temp_agent.run(prompt)
        print(f"Low temperature (0.1): {result1.output}")
        
        # High temperature response
        high_temp_agent = Agent(model=high_temp_model)
        result2 = await high_temp_agent.run(prompt)
        print(f"High temperature (0.9): {result2.output}")
        print()
        
    except Exception as e:
        print(f"Temperature comparison failed: {e}")
        print()
    
    # Example 3: Token limits
    print("=== Token Limit Example ===")
    
    limited_model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        settings={
            'max_tokens': 50,  # Very limited response
            'temperature': 0.5
        }
    )
    
    try:
        limited_agent = Agent(model=limited_model)
        result = await limited_agent.run("Explain quantum computing in detail")
        print(f"Limited tokens response: {result.output}")
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        print(f"Token limit example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
