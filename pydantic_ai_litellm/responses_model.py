"""LiteLLM Responses API model implementation for the agent framework pydantic-ai.

Some OpenAI models (e.g. the newer reasoning-capable `gpt-5.x` line) reject function
tools on `/v1/chat/completions` once reasoning is involved, and require
`/v1/responses` instead. `LiteLLMResponsesModel` targets that endpoint via
`litellm.aresponses`, as a sibling of `LiteLLMModel` rather than a hidden branch
inside it -- the two APIs have different request/response shapes (`input` items vs.
`messages`, `output` items vs. `choices`), so keeping them separate keeps the
Chat Completions path (used by the other 100+ LiteLLM providers) simple and stable.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from typing_extensions import assert_never

from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, get_user_agent

try:
    from litellm import aresponses
except ImportError as _import_error:
    raise ImportError(
        'Please install `litellm` to use the LiteLLM Responses model'
    ) from _import_error

__all__ = (
    'LiteLLMResponsesModel',
    'LiteLLMResponsesModelSettings',
)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Responses API item that may come back as a dict or a model object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class LiteLLMResponsesModelSettings(ModelSettings, total=False):
    """Settings used for a LiteLLM Responses API model request."""

    # ALL FIELDS MUST BE `litellm_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    """API key for the model provider."""
    litellm_api_key: str

    """Base URL for the model provider."""
    litellm_api_base: str

    """Custom LLM provider name for LiteLLM."""
    litellm_custom_llm_provider: str

    """Additional metadata to pass to LiteLLM."""
    litellm_metadata: dict[str, Any]

    """Whether the provider stores the response server-side (OpenAI's `store`).

    Pair `store=False` with `litellm_include=['reasoning.encrypted_content']` to replay
    reasoning statelessly across turns.
    """
    litellm_store: bool

    """Extra `include` values for the Responses API, e.g. `['reasoning.encrypted_content']`
    so reasoning models return encrypted reasoning; without it, reasoning items are dropped
    from replayed history rather than sent malformed."""
    litellm_include: list[str]


@dataclass(init=False)
class LiteLLMResponsesModel(Model):
    """A model that uses LiteLLM's Responses API (`aresponses`) to call various LLM providers.

    Use this instead of `LiteLLMModel` for models that require `/v1/responses` for
    function tools, such as newer reasoning-capable OpenAI models.
    See https://docs.litellm.ai/docs/providers for a list of supported providers.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    _model_name: str = field(repr=False)
    _api_key: str | None = field(default=None, repr=False)
    _api_base: str | None = field(default=None, repr=False)
    _custom_llm_provider: str | None = field(default=None, repr=False)
    _system: str = field(default='litellm', repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a LiteLLM Responses model.

        Args:
            model_name: The name of the model to use with LiteLLM (e.g., 'gpt-5.1', 'o3').
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            custom_llm_provider: Custom LLM provider name for LiteLLM. Use this if LiteLLM can't auto-detect the provider.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name
        self._api_key = api_key
        self._api_base = api_base
        self._custom_llm_provider = custom_llm_provider

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return self._api_base

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._response_create(
            messages, False, cast(LiteLLMResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._response_create(
            messages, True, cast(LiteLLMResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        yield await self._process_streamed_response(response, model_request_parameters)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    async def _response_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: LiteLLMResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        tools = self._get_tools(model_request_parameters)

        tool_choice: str | None = None
        if tools:
            if not model_request_parameters.allow_text_output:
                tool_choice = 'required'
            else:
                tool_choice = 'auto'

        input_items = await self._map_messages(messages)
        instructions = self._build_instructions(messages, model_request_parameters)

        # Prepare responses arguments
        response_kwargs: dict[str, Any] = {
            'model': self._model_name,
            'input': input_items,
            'stream': stream,
        }

        if instructions:
            response_kwargs['instructions'] = instructions

        # Add optional parameters from model settings
        if tools:
            response_kwargs['tools'] = tools
            if tool_choice:
                response_kwargs['tool_choice'] = tool_choice

        # Settings are checked with `is not None` rather than truthiness so that falsy-but-set
        # values (`temperature=0`, `parallel_tool_calls=False`, etc.) are still forwarded.
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            response_kwargs['parallel_tool_calls'] = parallel_tool_calls

        # The Responses API calls this `max_output_tokens`, unlike Chat Completions' `max_tokens`.
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            response_kwargs['max_output_tokens'] = max_tokens

        if (temperature := model_settings.get('temperature')) is not None:
            response_kwargs['temperature'] = temperature

        if (top_p := model_settings.get('top_p')) is not None:
            response_kwargs['top_p'] = top_p

        if (timeout := model_settings.get('timeout')) is not None:
            response_kwargs['timeout'] = timeout

        # `stop_sequences` and `seed` have no Responses API equivalent, so unlike
        # `LiteLLMModel._completion_create` they are intentionally not forwarded here.

        # Add LiteLLM-specific parameters
        api_key = model_settings.get('litellm_api_key') or self._api_key
        if api_key:
            response_kwargs['api_key'] = api_key

        api_base = model_settings.get('litellm_api_base') or self._api_base
        if api_base:
            response_kwargs['api_base'] = api_base

        custom_provider = model_settings.get('litellm_custom_llm_provider') or self._custom_llm_provider
        if custom_provider:
            response_kwargs['custom_llm_provider'] = custom_provider

        if (metadata := model_settings.get('litellm_metadata')) is not None:
            response_kwargs['metadata'] = metadata

        if (store := model_settings.get('litellm_store')) is not None:
            response_kwargs['store'] = store

        if (include := model_settings.get('litellm_include')) is not None:
            response_kwargs['include'] = include

        if (extra_headers := model_settings.get('extra_headers')) is not None:
            extra_headers = dict(extra_headers)
            extra_headers.setdefault('User-Agent', get_user_agent())
            response_kwargs['extra_headers'] = extra_headers

        if (extra_body := model_settings.get('extra_body')) is not None:
            response_kwargs['extra_body'] = extra_body

        try:
            return await aresponses(**response_kwargs)
        except Exception as e:
            # LiteLLM may raise various exceptions depending on the provider
            # We'll wrap them in ModelHTTPError if they look like HTTP errors
            if hasattr(e, 'status_code') and isinstance(e.status_code, int) and e.status_code >= 400:
                raise ModelHTTPError(
                    status_code=e.status_code,
                    model_name=self.model_name,
                    body=str(e)
                ) from e
            raise  # Re-raise other exceptions as-is

    def _process_response(self, response: Any) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        if not response.output:
            raise UnexpectedModelBehavior('No output returned from LiteLLM')

        items: list[ModelResponsePart] = []

        for output_item in response.output:
            item_type = _get_field(output_item, 'type')

            if item_type == 'message':
                for content_item in _get_field(output_item, 'content') or []:
                    if _get_field(content_item, 'type') == 'output_text':
                        text = _get_field(content_item, 'text')
                        if text:
                            items.append(TextPart(content=text))

            elif item_type == 'function_call':
                # Keep `call_id` (pairs with the output) and the item's own `id` separate:
                # reasoning models require both to be sent back on the next turn.
                item_id = _get_field(output_item, 'id')
                part = ToolCallPart(
                    tool_name=_get_field(output_item, 'name'),
                    args=_get_field(output_item, 'arguments'),
                    tool_call_id=_get_field(output_item, 'call_id') or item_id,
                )
                part.tool_call_id = _guard_tool_call_id(part)
                if item_id:
                    part.id = item_id
                    part.provider_name = self._system  # required whenever `id` is set
                items.append(part)

            elif item_type == 'reasoning':
                # Reasoning models (e.g. `gpt-5.x`, `o3`) return their chain-of-thought as a
                # dedicated `reasoning` output item, separate from `message`/`function_call`.
                # It must round-trip back through `_map_messages` on the next turn. Some
                # providers reject a `function_call` that isn't preceded by the `reasoning`
                # item that produced it.
                reasoning_id = _get_field(output_item, 'id')
                signature = _get_field(output_item, 'encrypted_content')
                summaries = _get_field(output_item, 'summary') or []
                if summaries:
                    for summary_item in summaries:
                        items.append(ThinkingPart(
                            content=_get_field(summary_item, 'text') or '',
                            id=reasoning_id,
                            signature=signature,
                            provider_name=self._system,
                        ))
                        # The signature belongs to the reasoning item as a whole, not to each
                        # summary. Only attach it to the first part so it isn't duplicated.
                        signature = None
                elif reasoning_id or signature:
                    items.append(ThinkingPart(
                        content='',
                        id=reasoning_id,
                        signature=signature,
                        provider_name=self._system,
                    ))

        # Map usage
        usage_obj = usage.RunUsage()
        if response.usage:
            usage_obj = usage.RunUsage(
                input_tokens=_get_field(response.usage, 'input_tokens', 0),
                output_tokens=_get_field(response.usage, 'output_tokens', 0),
            )

        # Get timestamp
        timestamp = _now_utc()
        if hasattr(response, 'created_at') and response.created_at:
            timestamp = datetime.fromtimestamp(response.created_at, tz=timestamp.tzinfo)

        return ModelResponse(
            items,
            usage=usage_obj,
            model_name=getattr(response, 'model', self._model_name),
            timestamp=timestamp,
            provider_response_id=getattr(response, 'id', None),
        )

    async def _process_streamed_response(
        self, response: Any, model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(
                'Streamed response ended without content or tool calls'
            )

        timestamp = _now_utc()
        first_response = _get_field(first_chunk, 'response')
        created_at = _get_field(first_response, 'created_at') if first_response is not None else None
        if created_at:
            timestamp = datetime.fromtimestamp(created_at, tz=timestamp.tzinfo)

        return LiteLLMResponsesStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=timestamp,
            model_request_parameters=model_request_parameters,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]]:
        """Convert tool definitions to Responses API format."""
        all_tools = model_request_parameters.function_tools + model_request_parameters.output_tools
        return [self._map_tool_definition(tool_def) for tool_def in all_tools]

    def _map_tool_definition(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Map a ToolDefinition to Responses API format.

        Unlike Chat Completions, the Responses API expects a flat tool dict --
        no nested `function` object.
        """
        return {
            'type': 'function',
            'name': tool_def.name,
            'description': tool_def.description or '',
            'parameters': tool_def.parameters_json_schema,
        }

    def _build_instructions(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> str | None:
        """Join agent instruction parts for the Responses API's dedicated `instructions` field.

        Unlike `LiteLLMModel`, there's no need to merge these into the input items the way
        `_merge_leading_system_messages` does for Chat Completions -- the Responses API already
        has a top-level `instructions` parameter for exactly this.
        """
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters)
        if not instruction_parts:
            return None
        return '\n\n'.join(part.content for part in instruction_parts)

    async def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Map pydantic_ai messages to Responses API `input` items."""
        input_items: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        input_items.append({
                            'type': 'message',
                            'role': 'system',
                            'content': part.content,
                        })
                    elif isinstance(part, UserPromptPart):
                        # For now, we'll handle simple string content
                        # More complex content (images, etc.) would need additional handling
                        content = part.content
                        if isinstance(content, str):
                            input_items.append({
                                'type': 'message',
                                'role': 'user',
                                'content': content,
                            })
                        else:
                            # For complex content, we'll convert to string for now
                            # In a full implementation, we'd handle images, files, etc.
                            content_str = ' '.join(str(item) for item in content if item)
                            if content_str:
                                input_items.append({
                                    'type': 'message',
                                    'role': 'user',
                                    'content': content_str,
                                })
                    elif isinstance(part, ToolReturnPart):
                        input_items.append({
                            'type': 'function_call_output',
                            'call_id': _guard_tool_call_id(t=part),
                            'output': part.model_response_str(),
                        })
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            input_items.append({
                                'type': 'message',
                                'role': 'user',
                                'content': part.model_response(),
                            })
                        else:
                            input_items.append({
                                'type': 'function_call_output',
                                'call_id': _guard_tool_call_id(t=part),
                                'output': part.model_response(),
                            })
                    else:
                        assert_never(part)

            elif isinstance(message, ModelResponse):
                # Reasoning items whose `id` we've already appended to `input_items` in this
                # message, so multiple `ThinkingPart`s for the same item (one per summary)
                # merge back into a single `reasoning` item instead of duplicating it.
                reasoning_items: dict[str, dict[str, Any]] = {}

                for part in message.parts:
                    if isinstance(part, TextPart):
                        input_items.append({
                            'type': 'message',
                            'role': 'assistant',
                            'content': part.content,
                        })
                    elif isinstance(part, ToolCallPart):
                        function_call_item: dict[str, Any] = {
                            'type': 'function_call',
                            'call_id': _guard_tool_call_id(t=part),
                            'name': part.tool_name,
                            'arguments': part.args_as_json_str(),
                        }
                        # Reasoning models require the function_call's own `id` alongside
                        # `call_id`. The id is only meaningful to the provider that issued it.
                        if part.id and part.provider_name == self._system:
                            function_call_item['id'] = part.id
                        input_items.append(function_call_item)
                    elif isinstance(part, ThinkingPart):
                        if part.provider_name != self._system or not part.id:
                            continue

                        reasoning_item = reasoning_items.get(part.id)
                        if reasoning_item is None:
                            # Only replay reasoning that carries its `encrypted_content`
                            # (`signature`); a bare reasoning id would 400 on the next turn since
                            # this model doesn't use `previous_response_id`. Enable via
                            # `litellm_include=['reasoning.encrypted_content']` + `litellm_store=False`.
                            if not part.signature:
                                continue
                            reasoning_item = {
                                'type': 'reasoning',
                                'id': part.id,
                                'summary': [],
                                'encrypted_content': part.signature,
                            }
                            reasoning_items[part.id] = reasoning_item
                            input_items.append(reasoning_item)

                        if part.content:
                            reasoning_item['summary'].append({'type': 'summary_text', 'text': part.content})
                    else:
                        # Handle other part types as needed
                        pass
            else:
                assert_never(message)

        return input_items


@dataclass
class LiteLLMResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for LiteLLM Responses API models."""

    _model_name: str
    _response: Any
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            event_type = _get_field(chunk, 'type')

            # Handle text content
            if event_type == 'response.output_text.delta':
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=_get_field(chunk, 'item_id'),
                    content=_get_field(chunk, 'delta'),
                ):
                    yield event

            # Handle the start of a tool call
            elif event_type == 'response.output_item.added':
                item = _get_field(chunk, 'item')
                if _get_field(item, 'type') == 'function_call':
                    # Emit a full part (not a delta) so we can carry the `id`; `handle_tool_call_delta`
                    # can't. Later `function_call_arguments.delta` events append to it.
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=_get_field(chunk, 'output_index'),
                        tool_name=_get_field(item, 'name'),
                        args=_get_field(item, 'arguments') or None,
                        tool_call_id=_get_field(item, 'call_id'),
                        id=_get_field(item, 'id'),
                        provider_name=self.provider_name,
                    )

            # Handle tool call argument deltas
            elif event_type == 'response.function_call_arguments.delta':
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=_get_field(chunk, 'output_index'),
                    tool_name=None,
                    args=_get_field(chunk, 'delta'),
                    tool_call_id=None,
                )
                if maybe_event is not None:
                    yield maybe_event

            # Handle reasoning summary text deltas
            elif event_type == 'response.reasoning_summary_text.delta':
                item_id = _get_field(chunk, 'item_id')
                summary_index = _get_field(chunk, 'summary_index', 0)
                # Same vendor id as summary index 0 so a single-summary reasoning item
                # accumulates into one ThinkingPart instead of a new part per delta.
                vendor_part_id = item_id if summary_index == 0 else f'{item_id}-{summary_index}'
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_part_id,
                    content=_get_field(chunk, 'delta'),
                    id=item_id,
                    provider_name=self.provider_name,
                ):
                    yield event

            # The reasoning item's signature is only available once it's finished streaming
            elif event_type == 'response.output_item.done':
                item = _get_field(chunk, 'item')
                if _get_field(item, 'type') == 'reasoning':
                    item_id = _get_field(item, 'id')
                    if signature := _get_field(item, 'encrypted_content'):
                        for event in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=item_id,
                            id=item_id,
                            signature=signature,
                            provider_name=self.provider_name,
                        ):
                            yield event

            # Update usage -- the Responses API only reports final usage once, on completion
            elif event_type == 'response.completed':
                response_obj = _get_field(chunk, 'response')
                usage_obj = _get_field(response_obj, 'usage')
                if usage_obj is not None:
                    self._usage += usage.RunUsage(
                        input_tokens=_get_field(usage_obj, 'input_tokens', 0),
                        output_tokens=_get_field(usage_obj, 'output_tokens', 0),
                    )

    @property
    def provider_url(self) -> str | None:
        """Get the provider base URL."""
        return None

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return 'litellm'
