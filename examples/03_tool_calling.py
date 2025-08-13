#!/usr/bin/env python3
"""
Tool Calling Example - Using functions as tools with pydantic-ai-litellm
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

def get_weather(location: str) -> str:
    """Get weather for a location."""
    # This is a mock function - in reality you'd call a weather API
    return f"It's sunny in {location}"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Using eval is dangerous in production - this is just for demo
        # In real code, use a proper math expression parser
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        else:
            return "Invalid characters in expression"
    except Exception as e:
        return f"Error: {e}"

async def main():
    """Example showing tool calling functionality."""
    
    model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    agent = Agent(model=model, tools=[get_weather, calculate])

    try:
        # Test weather tool
        result = await agent.run("What's the weather in Paris?")
        print(f"Weather query result: {result.output}")
        print(f"Usage: {result.usage()}")
        print()
        
        # Test calculation tool
        result = await agent.run("What is 25 * 4 + 10?")
        print(f"Calculation result: {result.output}")
        print(f"Usage: {result.usage()}")
        print()
        
        # Test multiple tools in one conversation
        result = await agent.run("Calculate 15 + 27 and then tell me the weather in Tokyo")
        print(f"Multi-tool result: {result.output}")
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        print(f"Tool calling failed: {e}")
        print("Make sure you have a valid API key for a model that supports tool calling")

if __name__ == "__main__":
    asyncio.run(main())
