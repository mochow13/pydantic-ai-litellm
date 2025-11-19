# Pydantic AI LiteLLM

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pydantic-ai-litellm?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pydantic-ai-litellm)

A LiteLLM model integration for the [Pydantic AI](https://ai.pydantic.dev/) framework, enabling access to 100+ LLM providers through a unified interface.

## Features

- **Universal LLM Access**: Connect to 100+ LLM providers (OpenAI, Anthropic, Cohere, Bedrock, Azure, and many more) via LiteLLM
- **Full Pydantic AI Integration**: Complete support for tool calling, streaming, structured outputs, and all Pydantic AI features
- **Type Safety**: Fully typed with comprehensive type hints
- **Async/Await Support**: Built for modern async Python applications
- **Flexible Configuration**: Support for custom API endpoints, headers, and provider-specific settings

## Installation

```bash
pip install pydantic-ai-litellm
```

## Quick Start

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

# Initialize with any LiteLLM-supported model
model = LiteLLMModel(
    model_name="gpt-4",  # or claude-3-opus, gemini-pro, etc.
    api_key="your-api-key"  # will also check environment variables
)

# Create an agent
agent = Agent(model=model)

# Run inference
async def main():
    result = await agent.run("What is the capital of France?")
    print(result.output)

asyncio.run(main())
```

## Supported Providers

This library supports all providers available through LiteLLM, including:

- **OpenAI**: GPT-4, GPT-3.5, o1, etc.
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro, Gemini Flash
- **AWS Bedrock**: Claude, Titan, Cohere models
- **Azure OpenAI**: All Azure-hosted models
- **Cohere**: Command, Command R+
- **Mistral AI**: Mistral 7B, 8x7B, Large
- **And 90+ more providers**

See the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for the complete list.

## Advanced Usage

### Custom API Endpoints

```python
model = LiteLLMModel(
    model_name="custom-model",
    api_base="https://your-custom-endpoint.com/v1",
    api_key="your-api-key",
    custom_llm_provider="openai"  # specify provider format
)
```

### Tool Calling

```python
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"It's sunny in {location}"

model = LiteLLMModel("gpt-4")
agent = Agent(model=model, tools=[get_weather])

result = await agent.run("What's the weather in Paris?")
```

### Streaming

```python
async with agent.run_stream("Write a poem about AI") as stream:
    async for text in stream.stream_text(delta=True):
        print(text, end="", flush=True)
```

### Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

agent = Agent(model=model, output_type=Person)
result = await agent.run("Generate a person profile")
print(result.output.name)  # Typed as Person
```

## Configuration

You can configure the model with various settings:

```python
from pydantic_ai_litellm import LiteLLMModelSettings

settings: LiteLLMModelSettings = {
    'temperature': 0.7,
    'max_tokens': 1000,
    'litellm_api_key': 'your-key',
    'litellm_api_base': 'https://custom-endpoint.com',
    'extra_headers': {'Custom-Header': 'value'}
}

model = LiteLLMModel("gpt-4", settings=settings)
```

## Requirements

- Python 3.13+
- `pydantic-ai-slim>=0.6.2`
- `litellm>=1.75.5`

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see LICENSE file for details.

## Examples

See the `examples/` directory for complete working examples:

- **Quick Start** (`examples/01_quick_start.py`) - Basic usage
- **Custom Endpoints** (`examples/02_custom_endpoints.py`) - Using custom API endpoints  
- **Tool Calling** (`examples/03_tool_calling.py`) - Functions as AI tools
- **Streaming** (`examples/04_streaming.py`) - Real-time text streaming
- **Structured Output** (`examples/05_structured_output.py`) - Typed responses with Pydantic
- **Configuration** (`examples/06_configuration.py`) - Model settings and parameters

Each example includes error handling and can be run independently with the appropriate API keys.

## Links

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [GitHub Repository](https://github.com/mochow13/pydantic-ai-litellm)
