# Pydantic AI LiteLLM Examples

This directory contains working examples demonstrating the features of `pydantic-ai-litellm`.

## Running the Examples

Before running any examples, make sure you have:

1. Installed the package: `pip install pydantic-ai-litellm`
2. Set up your API keys as environment variables

## Examples Overview

### 1. Quick Start (`01_quick_start.py`)
Basic usage example showing how to create a model and run a simple query.

**Required environment variables:**
- `GEMINI_API_KEY`: Your Google Gemini API key

```bash
python examples/01_quick_start.py
```

### 2. Custom Endpoints (`02_custom_endpoints.py`)
Demonstrates how to use custom API endpoints and proxies.

**Required environment variables:**
- `ANTHROPIC_MODEL`: Model name (e.g., "claude-3-sonnet-20240229")
- `LITELLM_API_BASE`: Custom API base URL
- `LITELLM_API_KEY`: API key for the custom endpoint

```bash
python examples/02_custom_endpoints.py
```

### 3. Tool Calling (`03_tool_calling.py`)
Shows how to use functions as tools that the AI can call.

**Required environment variables:**
- `OPENAI_API_KEY`: Your OpenAI API key

```bash
python examples/03_tool_calling.py
```

### 4. Streaming (`04_streaming.py`)
Demonstrates real-time text streaming for long responses.

**Required environment variables:**
- `GEMINI_API_KEY`: Your Google Gemini API key

```bash
python examples/04_streaming.py
```

### 5. Structured Output (`05_structured_output.py`)
Uses Pydantic models to get typed, structured responses from the AI.

**Required environment variables:**
- `GEMINI_API_KEY`: Your Google Gemini API key

```bash
python examples/05_structured_output.py
```

### 6. Configuration (`06_configuration.py`)
Shows different model configuration options like temperature, token limits, etc.

**Required environment variables:**
- `GEMINI_API_KEY`: Your Google Gemini API key

```bash
python examples/06_configuration.py
```

## Setting Up Environment Variables

Create a `.env` file in your project root or set these variables in your shell:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key-here"

# Google Gemini
export GEMINI_API_KEY="your-gemini-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Custom endpoints (if using)
export LITELLM_API_BASE="https://your-custom-endpoint.com"
export LITELLM_API_KEY="your-custom-key-here"
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"
```

## Common Issues

### Missing API Keys
If you get authentication errors, make sure your API keys are properly set as environment variables.

### Model Not Available
Some examples use specific models. If you get a model not found error, try substituting with a model you have access to.

### Rate Limits
Some providers have rate limits. If you hit them, wait a moment and try again.

## Extending the Examples

These examples are designed to be educational and can be modified to suit your needs. Feel free to:

- Change the model names to ones you have access to
- Modify the prompts and functions
- Add your own Pydantic models for structured output
- Experiment with different configuration settings

## Need Help?

- Check the main README.md for more information
- Review the [Pydantic AI documentation](https://ai.pydantic.dev/)
- See the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for supported models
