#!/usr/bin/env python3
"""
Test script for LiteLLM integration with PydanticAI
This demonstrates actual LLM functionality using various providers through LiteLLM
"""

import asyncio
import os
import pytest
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.asyncio
async def test_basic_completion():
    """Test basic completion functionality"""
    print("=== Testing Basic Completion ===")

    try:
        model = LiteLLMModel(
            model_name=os.getenv('DEEPSEEK_MODEL'),
            api_key=os.getenv('LITELLM_API_KEY'),
            api_base=os.getenv('LITELLM_API_BASE'),
        )
        agent = Agent(model=model)
        
        print(f"\nTesting {model.model_name}...")
        
        result = await agent.run("What is 2+2? Answer with just the number.")
        print(f"Result: {result.output}")
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        print(f"Failed to test {model.model_name}: {str(e)}")
    else:
        print("No models were successfully tested. Make sure you have API keys set up.")


@pytest.mark.asyncio
async def test_streaming():
    """Test streaming functionality"""
    print("\n=== Testing Streaming ===")
    
    try:
        # Use OpenAI if available, fallback to others
        model = LiteLLMModel(
            model_name=os.getenv('ANTHROPIC_MODEL'),
            api_key=os.getenv('LITELLM_API_KEY'),
            api_base=os.getenv('LITELLM_API_BASE'),
        )
        agent = Agent(model=model)

        print("Streaming response:")
        async with agent.run_stream("Write a sonnet about sonnet.") as stream:
            async for text in stream.stream_text(delta=True):
                print(text, end='', flush=True)
        
        print("\nStreaming completed.")
        
    except Exception as e:
        print(f"Streaming test failed: {str(e)}")


@pytest.mark.asyncio
async def test_structured_output():
    """Test structured output with Pydantic models"""
    print("\n=== Testing Structured Output ===")
    
    from pydantic import BaseModel
    from typing import List
    
    class CodeReview(BaseModel):
        overall_score: int  # 1-10
        strengths: List[str]
        suggestions: List[str]
        
    try:
        # Use a model that supports tools/function calling
        model = LiteLLMModel(
            model_name=os.getenv('OPENAI_MODEL'),
            api_key=os.getenv('LITELLM_API_KEY'),
            api_base=os.getenv('LITELLM_API_BASE'),
        )
        agent = Agent(model=model, output_type=CodeReview)
        
        code_snippet = """
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n-1) + fibonacci(n-2)
        """
        
        result = await agent.run(
            f"Review this code and provide feedback:\n{code_snippet}"
        )
        
        print(f"Overall Score: {result.output.overall_score}/10")
        print(f"Strengths: {result.output.strengths}")
        print(f"Suggestions: {result.output.suggestions}")
        
    except Exception as e:
        print(f"Structured output test failed: {str(e)}")


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@pytest.mark.asyncio
async def test_tool_calling():
    """Test tool calling functionality"""
    print("\n=== Testing Tool Calling ===")

    # Use a model that supports tools/function calling
    model = LiteLLMModel(
        model_name=os.getenv('OPENAI_MODEL'),
        api_key=os.getenv('LITELLM_API_KEY'),
        api_base=os.getenv('LITELLM_API_BASE'),
    )

    agent = Agent(model=model, tools=[multiply])

    try:
        result = await agent.run("What is 2129731231494*208977523231?")

        print(f"Result: {result.output}")
        print(f"Usage: {result.usage()}")
        print(f"Messages: {result.all_messages()}")
        print(f"Test tool calling completed.")

    except Exception as e:
        print(f"Tool calling test failed: {str(e)}")

async def main():
    """Run all tests"""
    print("PydanticAI LiteLLM Integration Test Suite")
    print("=" * 50)
    
    await test_basic_completion()
    await test_streaming()
    await test_structured_output()
    await test_tool_calling()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
