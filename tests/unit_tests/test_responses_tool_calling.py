"""
Test suite for tool calling capabilities of LiteLLMResponsesModel.

Mirrors `test_tool_calling.py`, adapted for the Responses API's request/response shape:
`aresponses(input=..., tools=[{'type': 'function', 'name': ..., ...}])` instead of
`acompletion(messages=..., tools=[{'type': 'function', 'function': {...}}])`, and a
response `.output` list of typed items instead of `.choices[0].message`.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import ModelRequest, ToolCallPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMResponsesModel


class MockLiteLLMResponsesResponse:
    """Mock response object for LiteLLM's Responses API (`aresponses`)."""

    def __init__(self, text: str = None, tool_calls: List[Dict] = None):
        self.output = []

        if text is not None:
            content_item = Mock()
            content_item.type = 'output_text'
            content_item.text = text

            message_item = Mock()
            message_item.type = 'message'
            message_item.content = [content_item]
            self.output.append(message_item)

        for call in tool_calls or []:
            call_item = Mock()
            call_item.type = 'function_call'
            call_item.call_id = call['call_id']
            call_item.name = call['name']
            call_item.arguments = call['arguments']
            self.output.append(call_item)

        self.usage = Mock()
        self.usage.input_tokens = 10
        self.usage.output_tokens = 20
        self.usage.total_tokens = 30

        self.model = "test-model"
        self.id = "test-response-id"
        self.created_at = 1640995200


class TestResponsesToolCalling:
    """Test cases for LiteLLMResponsesModel tool calling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = LiteLLMResponsesModel(
            model_name="gpt-5.1",
            api_key="test-key"
        )

        # Sample tool definition
        self.calculator_tool = ToolDefinition(
            name="calculator",
            description="Perform basic arithmetic operations",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        )

    def test_map_tool_definition(self):
        """Test mapping a tool definition to the Responses API's flat format."""
        result = self.model._map_tool_definition(self.calculator_tool)

        expected = {
            'type': 'function',
            'name': 'calculator',
            'description': 'Perform basic arithmetic operations',
            'parameters': {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }

        assert result == expected

    def test_get_tools(self):
        """Test getting tools from model parameters."""
        model_params = ModelRequestParameters(
            function_tools=[self.calculator_tool],
            output_tools=[],
            allow_text_output=True
        )

        tools = self.model._get_tools(model_params)

        assert len(tools) == 1
        assert tools[0]['name'] == 'calculator'

    @pytest.mark.asyncio
    @patch('pydantic_ai_litellm.responses_model.aresponses')
    async def test_completion_with_tools(self, mock_aresponses):
        """Test completion with tools and auto tool choice."""
        mock_response = MockLiteLLMResponsesResponse()
        mock_aresponses.return_value = mock_response

        model_params = ModelRequestParameters(
            function_tools=[self.calculator_tool],
            output_tools=[],
            allow_text_output=True
        )

        messages = [ModelRequest([UserPromptPart("Calculate 5 + 3")])]

        await self.model._response_create(
            messages=messages,
            stream=False,
            model_settings={},
            model_request_parameters=model_params
        )

        # Verify the call was made with correct parameters
        mock_aresponses.assert_called_once()
        call_args = mock_aresponses.call_args[1]

        assert call_args['model'] == 'gpt-5.1'
        assert 'input' in call_args
        assert len(call_args['tools']) == 1
        assert call_args['tools'][0]['name'] == 'calculator'
        assert call_args['tool_choice'] == 'auto'

    @pytest.mark.asyncio
    @patch('pydantic_ai_litellm.responses_model.aresponses')
    async def test_required_tool_choice(self, mock_aresponses):
        """Test completion with required tool choice (no text output allowed)."""
        mock_response = MockLiteLLMResponsesResponse()
        mock_aresponses.return_value = mock_response

        model_params = ModelRequestParameters(
            function_tools=[self.calculator_tool],
            output_tools=[],
            allow_text_output=False  # This should set tool_choice to 'required'
        )

        messages = [ModelRequest([UserPromptPart("Calculate something")])]

        await self.model._response_create(
            messages=messages,
            stream=False,
            model_settings={},
            model_request_parameters=model_params
        )

        call_args = mock_aresponses.call_args[1]
        assert call_args['tool_choice'] == 'required'

    def test_process_response_with_tool_call(self):
        """Test processing a response that contains a tool call."""
        mock_response = MockLiteLLMResponsesResponse(
            text="I'll calculate that for you.",
            tool_calls=[{'call_id': 'call_123', 'name': 'calculator', 'arguments': '{"operation": "add", "a": 5, "b": 3}'}]
        )

        result = self.model._process_response(mock_response)

        assert len(result.parts) == 2

        # Check text part
        text_part = next(part for part in result.parts if isinstance(part, TextPart))
        assert text_part.content == "I'll calculate that for you."

        # Check tool call part
        tool_call_part = next(part for part in result.parts if isinstance(part, ToolCallPart))
        assert tool_call_part.tool_name == "calculator"
        assert tool_call_part.args == '{"operation": "add", "a": 5, "b": 3}'
        assert tool_call_part.tool_call_id == "call_123"

    def test_process_response_text_only(self):
        """Test processing a response with only text content (no tool calls)."""
        mock_response = MockLiteLLMResponsesResponse(text="This is a simple text response.")

        result = self.model._process_response(mock_response)

        assert len(result.parts) == 1
        text_part = result.parts[0]
        assert isinstance(text_part, TextPart)
        assert text_part.content == "This is a simple text response."

    def test_no_tools_scenario(self):
        """Test behavior when no tools are provided."""
        model_params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True
        )

        tools = self.model._get_tools(model_params)
        assert tools == []

    def test_process_response_does_not_set_requests(self):
        """_process_response must not set requests — the pydantic-ai framework counts it."""
        mock_response = MockLiteLLMResponsesResponse(text="Hello")
        result = self.model._process_response(mock_response)
        assert result.usage.requests == 0
