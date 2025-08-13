"""
Test suite for tool calling capabilities of LiteLLMModel.

This test suite covers the core tool calling functionality including
tool definition mapping, tool choice handling, and response processing.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import ModelRequest, ToolCallPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMModel


class MockLiteLLMResponse:
    """Mock response object for LiteLLM completion."""
    
    def __init__(self, content: str = None, tool_calls: List[Dict] = None):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].message.tool_calls = tool_calls or []
        self.choices[0].finish_reason = 'tool_calls' if tool_calls else 'stop'
        
        self.usage = Mock()
        self.usage.prompt_tokens = 10
        self.usage.completion_tokens = 20
        self.usage.total_tokens = 30
        
        self.model = "test-model"
        self.id = "test-response-id"
        self.created = 1640995200


class TestToolCalling:
    """Test cases for tool calling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LiteLLMModel(
            model_name="gpt-4",
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
        """Test mapping a tool definition to LiteLLM format."""
        result = self.model._map_tool_definition(self.calculator_tool)
        
        expected = {
            'type': 'function',
            'function': {
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
        assert tools[0]['function']['name'] == 'calculator'

    @pytest.mark.asyncio
    @patch('pydantic_ai_litellm.litellm_model.acompletion')
    async def test_completion_with_tools(self, mock_acompletion):
        """Test completion with tools and auto tool choice."""
        mock_response = MockLiteLLMResponse()
        mock_acompletion.return_value = mock_response
        
        model_params = ModelRequestParameters(
            function_tools=[self.calculator_tool],
            output_tools=[],
            allow_text_output=True
        )
        
        messages = [ModelRequest([UserPromptPart("Calculate 5 + 3")])]
        
        await self.model._completion_create(
            messages=messages,
            stream=False,
            model_settings={},
            model_request_parameters=model_params
        )
        
        # Verify the call was made with correct parameters
        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args[1]
        
        assert call_args['model'] == 'gpt-4'
        assert len(call_args['tools']) == 1
        assert call_args['tools'][0]['function']['name'] == 'calculator'
        assert call_args['tool_choice'] == 'auto'

    @pytest.mark.asyncio
    @patch('pydantic_ai_litellm.litellm_model.acompletion')
    async def test_required_tool_choice(self, mock_acompletion):
        """Test completion with required tool choice (no text output allowed)."""
        mock_response = MockLiteLLMResponse()
        mock_acompletion.return_value = mock_response
        
        model_params = ModelRequestParameters(
            function_tools=[self.calculator_tool],
            output_tools=[],
            allow_text_output=False  # This should set tool_choice to 'required'
        )
        
        messages = [ModelRequest([UserPromptPart("Calculate something")])]
        
        await self.model._completion_create(
            messages=messages,
            stream=False,
            model_settings={},
            model_request_parameters=model_params
        )
        
        call_args = mock_acompletion.call_args[1]
        assert call_args['tool_choice'] == 'required'

    def test_process_response_with_tool_call(self):
        """Test processing a response that contains a tool call."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "calculator"
        mock_tool_call.function.arguments = '{"operation": "add", "a": 5, "b": 3}'
        
        mock_response = MockLiteLLMResponse(
            content="I'll calculate that for you.",
            tool_calls=[mock_tool_call]
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
        mock_response = MockLiteLLMResponse(content="This is a simple text response.")
        
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
