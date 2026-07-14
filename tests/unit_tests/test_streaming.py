"""Regression tests for LiteLLMStreamedResponse against pydantic-ai 2.9.

pydantic-ai 2.9 changed `ModelPartsManager.handle_text_delta` to return an
`Iterator[ModelResponseStreamEvent]` (zero or more events) instead of
`ModelResponseStreamEvent | None`. `_get_event_iterator` used to treat it as
optional (`if maybe_event is not None: yield maybe_event`) -- since an
iterator is never None, that check always passed and yielded the raw
iterator object instead of the events inside it.

`StreamedResponse.provider_url` is also an `@abstractmethod` on the base
class, so `LiteLLMStreamedResponse` couldn't be instantiated at all without
implementing it.
"""

from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMModel


def _make_chunk(
    *,
    content: str | None = None,
    tool_call_deltas: list[Mock] | None = None,
    usage: Mock | None = None,
    created: int | None = None,
) -> Mock:
    """Build a fake chunk shaped like a LiteLLM/OpenAI streaming response chunk."""
    chunk = Mock()
    chunk.created = created
    chunk.usage = usage

    delta = Mock()
    delta.content = content
    delta.tool_calls = tool_call_deltas or []

    choice = Mock()
    choice.delta = delta
    chunk.choices = [choice]

    return chunk


def _make_tool_call_delta(*, tool_call_id: str, name: str, arguments: str) -> Mock:
    tool_call = Mock()
    tool_call.id = tool_call_id
    tool_call.function = Mock()
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


async def _chunks(*items: Mock) -> AsyncIterator[Mock]:
    for item in items:
        yield item


def _params() -> ModelRequestParameters:
    return ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)


class TestStreaming:
    def setup_method(self):
        self.model = LiteLLMModel(model_name="gpt-4", api_key="test-key")

    @pytest.mark.asyncio
    async def test_streamed_text_response_yields_text_events(self):
        """The actual regression: the second content chunk must produce a real
        PartDeltaEvent, not the raw handle_text_delta() iterator object yielded
        as if it were a single event.

        Filtering by event type rather than asserting a total count on purpose --
        pydantic-ai also emits its own FinalResultEvent/PartEndEvent around these,
        which aren't specific to this bug and could shift independently of it.
        """
        chunks = _chunks(
            _make_chunk(content="Hello", created=1_700_000_000),
            _make_chunk(content=", world!"),
        )

        streamed = await self.model._process_streamed_response(chunks, _params())
        events = [event async for event in streamed]

        start_events = [e for e in events if isinstance(e, PartStartEvent)]
        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]

        assert len(start_events) == 1
        assert isinstance(start_events[0].part, TextPart)
        assert start_events[0].part.content == "Hello"

        assert len(delta_events) == 1
        assert isinstance(delta_events[0].delta, TextPartDelta)
        assert delta_events[0].delta.content_delta == ", world!"

    @pytest.mark.asyncio
    async def test_streamed_tool_call_response_still_works(self):
        """Guards that fixing the text-delta bug didn't disturb tool-call handling
        -- handle_tool_call_delta genuinely returns Event | None, unchanged."""
        tool_call_delta = _make_tool_call_delta(
            tool_call_id="call_1", name="calculator", arguments='{"a": 1, "b": 2}'
        )
        chunks = _chunks(_make_chunk(tool_call_deltas=[tool_call_delta], created=1_700_000_000))

        streamed = await self.model._process_streamed_response(chunks, _params())
        events = [event async for event in streamed]

        start_events = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(start_events) == 1
        assert isinstance(start_events[0].part, ToolCallPart)
        assert start_events[0].part.tool_name == "calculator"
        assert start_events[0].part.args == '{"a": 1, "b": 2}'

    @pytest.mark.asyncio
    async def test_streamed_response_usage_is_accumulated(self):
        usage_chunk = Mock(prompt_tokens=10, completion_tokens=5)
        chunks = _chunks(
            _make_chunk(content="Hi", usage=usage_chunk, created=1_700_000_000),
        )

        streamed = await self.model._process_streamed_response(chunks, _params())
        _ = [event async for event in streamed]

        assert streamed.usage.input_tokens == 10
        assert streamed.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_provider_url_does_not_raise(self):
        """provider_url is an abstractmethod on StreamedResponse -- constructing
        LiteLLMStreamedResponse at all would fail without an implementation."""
        chunks = _chunks(_make_chunk(content="hi", created=1_700_000_000))

        streamed = await self.model._process_streamed_response(chunks, _params())

        assert streamed.provider_url is None
