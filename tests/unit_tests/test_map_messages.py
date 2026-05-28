"""Tests for LiteLLMModel message mapping and instruction handling."""

import pytest

from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMModel


class TestMapMessages:
    """Tests for _map_messages instruction handling."""

    def setup_method(self):
        self.model = LiteLLMModel(model_name="gpt-4", api_key="test-key")

    @pytest.mark.asyncio
    async def test_instruction_parts_inserted_before_user(self):
        """Instruction parts are inserted before the first non-system message."""
        messages = [ModelRequest([UserPromptPart("Hello")])]
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
            instruction_parts=[InstructionPart(content="Be helpful.")],
        )

        result = await self.model._map_messages(messages, params)

        assert len(result) == 2
        assert result[0] == {'role': 'system', 'content': 'Be helpful.'}
        assert result[1] == {'role': 'user', 'content': 'Hello'}

    @pytest.mark.asyncio
    async def test_multiple_instruction_parts_merged_with_system_prompt(self):
        """Multiple instruction parts and SystemPromptPart merge into one system message."""
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

        result = await self.model._map_messages(messages, params)

        assert len(result) == 2
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == 'Base prompt.\n\nInstruction A.\n\nInstruction B.'
        assert result[1] == {'role': 'user', 'content': 'Hello'}

    @pytest.mark.asyncio
    async def test_no_instruction_parts(self):
        """No extra system message when instruction parts are absent."""
        messages = [ModelRequest([UserPromptPart("Hello")])]
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
        )

        result = await self.model._map_messages(messages, params)

        assert len(result) == 1
        assert result[0] == {'role': 'user', 'content': 'Hello'}
