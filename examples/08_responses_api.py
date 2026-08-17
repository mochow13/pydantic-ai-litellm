#!/usr/bin/env python3
"""
Responses API Example - Tool calling with reasoning-capable OpenAI models

Some OpenAI models (e.g. the `gpt-5.x` line) reject function tools on
`/v1/chat/completions` once reasoning is involved, and require `/v1/responses`
instead -- see `examples/03_tool_calling.py` for the error this produces with
`LiteLLMModel`. `LiteLLMResponsesModel` targets `/v1/responses` instead, so tool
calling works with those models.
"""

import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMResponsesModel

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
    """Example showing tool calling with a reasoning model via the Responses API."""

    model = LiteLLMResponsesModel(
        model_name=os.getenv("MODEL_NAME", "gpt-5.6-luna"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = Agent(model=model, tools=[get_weather, calculate])

    try:
        result = await agent.run("What's the weather in Paris, and what is 25 * 4 + 10?")
        print(f"Result: {result.output}")
        print(f"Usage: {result.usage}")

    except Exception as e:
        print(f"Tool calling failed: {e}")
        print("Make sure you have a valid API key for a model that supports the Responses API")

if __name__ == "__main__":
    asyncio.run(main())
