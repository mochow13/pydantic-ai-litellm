"""Tests for LiteLLMResponsesModel message mapping and instruction handling.

Mirrors `test_map_messages.py`, but the Responses API has a dedicated top-level
`instructions` field, so instruction parts must NOT be merged into the `input` items the
way `LiteLLMModel._map_messages` merges them into leading system messages.
"""

import pytest

from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMResponsesModel


class TestResponsesMapMessages:
    """Tests for `_map_messages` / `_build_instructions`."""

    def setup_method(self):
        self.model = LiteLLMResponsesModel(model_name="gpt-5.1", api_key="test-key")

    @pytest.mark.asyncio
    async def test_instructions_kept_separate_from_input(self):
        """Instruction parts must not appear as input items -- they belong in the
        Responses API's dedicated `instructions` field instead."""
        messages = [ModelRequest([UserPromptPart("Hello")])]
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
            instruction_parts=[InstructionPart(content="Be helpful.")],
        )

        input_items = await self.model._map_messages(messages)
        instructions = self.model._build_instructions(messages, params)

        assert input_items == [{'type': 'message', 'role': 'user', 'content': 'Hello'}]
        assert instructions == "Be helpful."

    @pytest.mark.asyncio
    async def test_multiple_instruction_parts_joined_separately_from_system_prompt(self):
        """Multiple instruction parts are joined for `instructions`; a `SystemPromptPart`
        stays its own input item rather than being merged in, unlike Chat Completions."""
        messages = [
            ModelRequest([
                SystemPromptPart("Base prompt."),
                UserPromptPart("Hello"),
            ])
        ]
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
            instruction_parts=[
                InstructionPart(content="Instruction A."),
                InstructionPart(content="Instruction B."),
            ],
        )

        input_items = await self.model._map_messages(messages)
        instructions = self.model._build_instructions(messages, params)

        assert input_items == [
            {'type': 'message', 'role': 'system', 'content': 'Base prompt.'},
            {'type': 'message', 'role': 'user', 'content': 'Hello'},
        ]
        assert instructions == "Instruction A.\n\nInstruction B."

    def test_no_instructions_when_absent(self):
        """No `instructions` string when instruction parts are absent."""
        messages = [ModelRequest([UserPromptPart("Hello")])]
        params = ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)

        assert self.model._build_instructions(messages, params) is None

    @pytest.mark.asyncio
    async def test_tool_round_trip_mapped_to_function_call_items(self):
        """ToolCallPart/ToolReturnPart map to the Responses API's flat `function_call` /
        `function_call_output` items sharing a `call_id` -- not nested under one assistant
        message the way Chat Completions nests `tool_calls`."""
        messages = [
            ModelRequest([UserPromptPart("What is 2+2?")]),
            ModelResponse([ToolCallPart(tool_name="calculator", args='{"a": 2, "b": 2}', tool_call_id="call_1")]),
            ModelRequest([ToolReturnPart(tool_name="calculator", content=4, tool_call_id="call_1")]),
        ]

        result = await self.model._map_messages(messages)

        assert result[0] == {'type': 'message', 'role': 'user', 'content': 'What is 2+2?'}
        assert result[1] == {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'calculator',
            'arguments': '{"a": 2, "b": 2}',
        }
        assert result[2]['type'] == 'function_call_output'
        assert result[2]['call_id'] == 'call_1'

    @pytest.mark.asyncio
    async def test_assistant_text_mapped_to_message_item(self):
        """A plain text ModelResponse maps to an assistant message item."""
        messages = [
            ModelRequest([UserPromptPart("Hi")]),
            ModelResponse([TextPart(content="Hello there!")]),
        ]

        result = await self.model._map_messages(messages)

        assert result[1] == {'type': 'message', 'role': 'assistant', 'content': 'Hello there!'}
