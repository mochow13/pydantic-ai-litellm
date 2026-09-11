"""Tests for LiteLLMResponsesStreamedResponse against LiteLLM's Responses API event stream.

Mirrors `test_streaming.py`, but the Responses API sends a different event shape than
Chat Completions: text and tool-call arguments arrive as separate `response.output_text.delta`
/ `response.function_call_arguments.delta` events (keyed by `item_id` / `output_index`)
instead of `choices[0].delta`, and usage is only reported once, on `response.completed`.
"""

from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_litellm import LiteLLMResponsesModel


def _make_event(event_type: str, **fields: object) -> Mock:
    """Build a fake event shaped like a LiteLLM Responses API streaming event.

    Always sets `response` (read by the initial peek for the response timestamp) so it
    defaults to `None` instead of leaking an auto-generated `Mock` attribute when unset.
    """
    event = Mock()
    event.type = event_type
    event.response = fields.pop('response', None)
    for name, value in fields.items():
        setattr(event, name, value)
    return event


def _make_function_call_item(*, call_id: str, name: str, arguments: str = '') -> Mock:
    item = Mock()
    item.type = 'function_call'
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_reasoning_item(*, item_id: str, encrypted_content: str | None = None) -> Mock:
    item = Mock()
    item.type = 'reasoning'
    item.id = item_id
    item.encrypted_content = encrypted_content
    return item


def _make_usage(*, input_tokens: int, output_tokens: int) -> Mock:
    usage = Mock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_response(*, created_at: int | None = None, usage: Mock | None = None) -> Mock:
    response = Mock()
    response.created_at = created_at
    response.usage = usage
    return response


async def _events(*items: Mock) -> AsyncIterator[Mock]:
    for item in items:
        yield item


def _params() -> ModelRequestParameters:
    return ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)


class TestResponsesStreaming:
    def setup_method(self):
        self.model = LiteLLMResponsesModel(model_name="gpt-5.1", api_key="test-key")

    @pytest.mark.asyncio
    async def test_streamed_text_response_yields_text_events(self):
        """Two `response.output_text.delta` events for the same `item_id` must produce a
        PartStartEvent followed by a real PartDeltaEvent, not two independent text parts."""
        events = _events(
            _make_event('response.output_text.delta', item_id='msg_1', delta='Hello'),
            _make_event('response.output_text.delta', item_id='msg_1', delta=', world!'),
        )

        streamed = await self.model._process_streamed_response(events, _params())
        received = [event async for event in streamed]

        start_events = [e for e in received if isinstance(e, PartStartEvent)]
        delta_events = [e for e in received if isinstance(e, PartDeltaEvent)]

        assert len(start_events) == 1
        assert isinstance(start_events[0].part, TextPart)
        assert start_events[0].part.content == "Hello"

        assert len(delta_events) == 1
        assert isinstance(delta_events[0].delta, TextPartDelta)
        assert delta_events[0].delta.content_delta == ", world!"

    @pytest.mark.asyncio
    async def test_streamed_tool_call_response_still_works(self):
        """A `response.output_item.added` (function_call) followed by
        `response.function_call_arguments.delta` events for the same `output_index` must
        start one ToolCallPart and then accumulate its arguments, not create separate calls."""
        events = _events(
            _make_event(
                'response.output_item.added',
                output_index=0,
                item=_make_function_call_item(call_id='call_1', name='calculator'),
            ),
            _make_event('response.function_call_arguments.delta', output_index=0, delta='{"a": 1,'),
            _make_event('response.function_call_arguments.delta', output_index=0, delta=' "b": 2}'),
        )

        streamed = await self.model._process_streamed_response(events, _params())
        received = [event async for event in streamed]

        start_events = [e for e in received if isinstance(e, PartStartEvent)]
        delta_events = [e for e in received if isinstance(e, PartDeltaEvent)]

        assert len(start_events) == 1
        assert isinstance(start_events[0].part, ToolCallPart)
        assert start_events[0].part.tool_name == "calculator"
        assert start_events[0].part.tool_call_id == "call_1"

        assert len(delta_events) == 2
        assert isinstance(delta_events[0].delta, ToolCallPartDelta)
        assert delta_events[0].delta.args_delta == '{"a": 1,'
        assert delta_events[1].delta.args_delta == ' "b": 2}'

    @pytest.mark.asyncio
    async def test_streamed_response_usage_is_accumulated(self):
        """Usage arrives once, on `response.completed`, unlike Chat Completions where every
        chunk can carry a usage update."""
        events = _events(
            _make_event('response.output_text.delta', item_id='msg_1', delta='Hi'),
            _make_event(
                'response.completed',
                response=_make_response(usage=_make_usage(input_tokens=10, output_tokens=5)),
            ),
        )

        streamed = await self.model._process_streamed_response(events, _params())
        _ = [event async for event in streamed]

        assert streamed.usage.input_tokens == 10
        assert streamed.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_streamed_reasoning_response_yields_thinking_events(self):
        """`response.reasoning_summary_text.delta` events for the same `item_id` must
        produce a PartStartEvent followed by a real PartDeltaEvent for a ThinkingPart --
        not be silently ignored the way an unhandled event type would be."""
        events = _events(
            _make_event(
                'response.reasoning_summary_text.delta', item_id='rs_1', summary_index=0, delta='Let me '
            ),
            _make_event(
                'response.reasoning_summary_text.delta', item_id='rs_1', summary_index=0, delta='think...'
            ),
        )

        streamed = await self.model._process_streamed_response(events, _params())
        received = [event async for event in streamed]

        start_events = [e for e in received if isinstance(e, PartStartEvent)]
        delta_events = [e for e in received if isinstance(e, PartDeltaEvent)]

        assert len(start_events) == 1
        assert isinstance(start_events[0].part, ThinkingPart)
        assert start_events[0].part.content == "Let me "
        assert start_events[0].part.id == "rs_1"

        assert len(delta_events) == 1
        assert isinstance(delta_events[0].delta, ThinkingPartDelta)
        assert delta_events[0].delta.content_delta == "think..."

    @pytest.mark.asyncio
    async def test_reasoning_signature_attached_on_output_item_done(self):
        """The reasoning item's `encrypted_content` (signature) is only available once
        streaming finishes, on `response.output_item.done` -- it must be merged into the
        ThinkingPart already started by the summary text deltas, not create a new part."""
        events = _events(
            _make_event(
                'response.reasoning_summary_text.delta', item_id='rs_1', summary_index=0, delta='Thinking...'
            ),
            _make_event(
                'response.output_item.done',
                item=_make_reasoning_item(item_id='rs_1', encrypted_content='enc-content'),
            ),
        )

        streamed = await self.model._process_streamed_response(events, _params())
        received = [event async for event in streamed]

        thinking_events = [
            e
            for e in received
            if (isinstance(e, PartStartEvent) and isinstance(e.part, ThinkingPart))
            or (isinstance(e, PartDeltaEvent) and isinstance(e.delta, ThinkingPartDelta))
        ]
        assert len(thinking_events) == 2

        signature_delta = thinking_events[1]
        assert isinstance(signature_delta, PartDeltaEvent)
        assert signature_delta.delta.signature_delta == "enc-content"

    @pytest.mark.asyncio
    async def test_provider_url_does_not_raise(self):
        """provider_url is an abstractmethod on StreamedResponse -- constructing
        LiteLLMResponsesStreamedResponse at all would fail without an implementation."""
        events = _events(_make_event('response.output_text.delta', item_id='msg_1', delta='hi'))

        streamed = await self.model._process_streamed_response(events, _params())

        assert streamed.provider_url is None
