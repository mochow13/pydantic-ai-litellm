#!/usr/bin/env python3
"""
Structured Output Example - Using Pydantic models for typed responses
"""

import asyncio
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

class CodeReview(BaseModel):
    overall_score: int  # 1-10
    strengths: list[str]
    suggestions: list[str]
    
class WeatherInfo(BaseModel):
    location: str
    temperature: int
    condition: str
    humidity: int

async def main():
    """Example showing structured output with Pydantic models."""
    
    model = LiteLLMModel(
        model_name="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Example 1: Generate a person profile
    print("=== Person Generation Example ===")
    agent = Agent(model=model, output_type=Person)
    
    try:
        result = await agent.run("Generate a person profile for a software engineer")
        person = result.output  # This is typed as Person
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
        print(f"Usage: {result.usage()}")
        print()
        
    except Exception as e:
        print(f"Person generation failed: {e}")
        print()
    
    # Example 2: Code review
    print("=== Code Review Example ===")
    review_agent = Agent(model=model, output_type=CodeReview)
    
    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """
    
    try:
        result = await review_agent.run(
            f"Review this code and provide feedback:\n{code_snippet}"
        )
        
        review = result.output  # This is typed as CodeReview
        print(f"Overall Score: {review.overall_score}/10")
        print(f"Strengths: {review.strengths}")
        print(f"Suggestions: {review.suggestions}")
        print(f"Usage: {result.usage()}")
        print()
        
    except Exception as e:
        print(f"Code review failed: {e}")
        print()
    
    # Example 3: Weather information
    print("=== Weather Info Example ===")
    weather_agent = Agent(model=model, output_type=WeatherInfo)
    
    try:
        result = await weather_agent.run(
            "Generate realistic weather information for Tokyo in summer"
        )
        
        weather = result.output  # This is typed as WeatherInfo
        print(f"Location: {weather.location}")
        print(f"Temperature: {weather.temperature}°C")
        print(f"Condition: {weather.condition}")
        print(f"Humidity: {weather.humidity}%")
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        print(f"Weather generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
