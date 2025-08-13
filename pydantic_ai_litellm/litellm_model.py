"""LiteLLM model implementation for the agent framework pydantic-ai."""

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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, get_user_agent

try:
    from litellm import acompletion
except ImportError as _import_error:
    raise ImportError(
        'Please install `litellm` to use the LiteLLM model'
    ) from _import_error

__all__ = (
    'LiteLLMModel',
    'LiteLLMModelSettings',
)


class LiteLLMModelSettings(ModelSettings, total=False):
    """Settings used for a LiteLLM model request."""

    # ALL FIELDS MUST BE `litellm_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    """API key for the model provider."""
    litellm_api_key: str

    """Base URL for the model provider."""
    litellm_api_base: str

    """Custom LLM provider name for LiteLLM."""
    litellm_custom_llm_provider: str

    """Additional metadata to pass to LiteLLM."""
    litellm_metadata: dict[str, Any]


@dataclass(init=False)
class LiteLLMModel(Model):
    """A model that uses LiteLLM to call various LLM providers.

    LiteLLM provides a unified interface to call 100+ LLMs using the same OpenAI format.
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
        """Initialize a LiteLLM model.

        Args:
            model_name: The name of the model to use with LiteLLM (e.g., 'gpt-4', 'claude-3-opus-20240229').
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
        response = await self._completion_create(
            messages, False, cast(LiteLLMModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        model_response.usage.requests = 1
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completion_create(
            messages, True, cast(LiteLLMModelSettings, model_settings or {}), model_request_parameters
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

    async def _completion_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: LiteLLMModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        tools = self._get_tools(model_request_parameters)
        
        tool_choice: str | None = None
        if tools:
            if not model_request_parameters.allow_text_output:
                tool_choice = 'required'
            else:
                tool_choice = 'auto'

        litellm_messages = await self._map_messages(messages)

        # Prepare completion arguments
        completion_kwargs: dict[str, Any] = {
            'model': self._model_name,
            'messages': litellm_messages,
            'stream': stream,
        }

        # Add optional parameters from model settings
        if tools:
            completion_kwargs['tools'] = tools
            if tool_choice:
                completion_kwargs['tool_choice'] = tool_choice

        if parallel_tool_calls := model_settings.get('parallel_tool_calls'):
            completion_kwargs['parallel_tool_calls'] = parallel_tool_calls

        if max_tokens := model_settings.get('max_tokens'):
            completion_kwargs['max_tokens'] = max_tokens

        if temperature := model_settings.get('temperature'):
            completion_kwargs['temperature'] = temperature

        if top_p := model_settings.get('top_p'):
            completion_kwargs['top_p'] = top_p

        if stop_sequences := model_settings.get('stop_sequences'):
            completion_kwargs['stop'] = stop_sequences

        if seed := model_settings.get('seed'):
            completion_kwargs['seed'] = seed

        if timeout := model_settings.get('timeout'):
            completion_kwargs['timeout'] = timeout

        # Add LiteLLM-specific parameters
        api_key = model_settings.get('litellm_api_key') or self._api_key
        if api_key:
            completion_kwargs['api_key'] = api_key

        api_base = model_settings.get('litellm_api_base') or self._api_base
        if api_base:
            completion_kwargs['api_base'] = api_base

        custom_provider = model_settings.get('litellm_custom_llm_provider') or self._custom_llm_provider
        if custom_provider:
            completion_kwargs['custom_llm_provider'] = custom_provider

        if metadata := model_settings.get('litellm_metadata'):
            completion_kwargs['metadata'] = metadata

        if extra_headers := model_settings.get('extra_headers'):
            extra_headers = dict(extra_headers)
            extra_headers.setdefault('User-Agent', get_user_agent())
            completion_kwargs['extra_headers'] = extra_headers

        if extra_body := model_settings.get('extra_body'):
            completion_kwargs['extra_body'] = extra_body

        try:
            return await acompletion(**completion_kwargs)
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
        if not response.choices:
            raise UnexpectedModelBehavior('No choices returned from LiteLLM')

        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        # Handle message content
        if choice.message and choice.message.content:
            items.append(TextPart(content=choice.message.content))

        # Handle tool calls
        if choice.message and choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if hasattr(tool_call, 'function'):
                    part = ToolCallPart(
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        tool_call_id=tool_call.id
                    )
                    part.tool_call_id = _guard_tool_call_id(part)
                    items.append(part)

        # Map usage
        usage_obj = usage.Usage()
        if response.usage:
            usage_obj = usage.Usage(
                request_tokens=getattr(response.usage, 'prompt_tokens', None),
                response_tokens=getattr(response.usage, 'completion_tokens', None),
                total_tokens=getattr(response.usage, 'total_tokens', None),
            )

        # Get timestamp
        timestamp = _now_utc()
        if hasattr(response, 'created') and response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timestamp.tzinfo)

        return ModelResponse(
            items,
            usage=usage_obj,
            model_name=getattr(response, 'model', self._model_name),
            timestamp=timestamp,
            vendor_id=getattr(response, 'id', None),
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
        if hasattr(first_chunk, 'created') and first_chunk.created:
            timestamp = datetime.fromtimestamp(first_chunk.created, tz=timestamp.tzinfo)

        return LiteLLMStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=timestamp,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM format (OpenAI-compatible)."""
        all_tools = model_request_parameters.function_tools + model_request_parameters.output_tools
        return [self._map_tool_definition(tool_def) for tool_def in all_tools]

    def _map_tool_definition(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Map a ToolDefinition to LiteLLM/OpenAI format."""
        return {
            'type': 'function',
            'function': {
                'name': tool_def.name,
                'description': tool_def.description or '',
                'parameters': tool_def.parameters_json_schema,
            },
        }

    async def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Map pydantic_ai messages to LiteLLM format (OpenAI-compatible)."""
        litellm_messages: list[dict[str, Any]] = []
        
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        litellm_messages.append({
                            'role': 'system',
                            'content': part.content,
                        })
                    elif isinstance(part, UserPromptPart):
                        # For now, we'll handle simple string content
                        # More complex content (images, etc.) would need additional handling
                        content = part.content
                        if isinstance(content, str):
                            litellm_messages.append({
                                'role': 'user',
                                'content': content,
                            })
                        else:
                            # For complex content, we'll convert to string for now
                            # In a full implementation, we'd handle images, files, etc.
                            content_str = ' '.join(str(item) for item in content if item)
                            if content_str:
                                litellm_messages.append({
                                    'role': 'user',
                                    'content': content_str,
                                })
                    elif isinstance(part, ToolReturnPart):
                        litellm_messages.append({
                            'role': 'tool',
                            'tool_call_id': _guard_tool_call_id(t=part),
                            'content': part.model_response_str(),
                        })
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            litellm_messages.append({
                                'role': 'user',
                                'content': part.model_response(),
                            })
                        else:
                            litellm_messages.append({
                                'role': 'tool',
                                'tool_call_id': _guard_tool_call_id(t=part),
                                'content': part.model_response(),
                            })
                    else:
                        assert_never(part)
                        
            elif isinstance(message, ModelResponse):
                message_content = None
                tool_calls = []
                
                for part in message.parts:
                    if isinstance(part, TextPart):
                        message_content = part.content
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append({
                            'id': _guard_tool_call_id(t=part),
                            'type': 'function',
                            'function': {
                                'name': part.tool_name,
                                'arguments': part.args_as_json_str(),
                            },
                        })
                    else:
                        # Handle other part types as needed
                        pass
                
                assistant_message: dict[str, Any] = {'role': 'assistant'}
                if message_content:
                    assistant_message['content'] = message_content
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                    
                litellm_messages.append(assistant_message)
            else:
                assert_never(message)

        # Add instructions as system message if present
        if instructions := self._get_instructions(messages):
            # Insert at the beginning
            litellm_messages.insert(0, {
                'role': 'system',
                'content': instructions,
            })

        return litellm_messages


@dataclass
class LiteLLMStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for LiteLLM models."""

    _model_name: str
    _response: Any
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            # Update usage if available
            if hasattr(chunk, 'usage') and chunk.usage:
                self._usage += usage.Usage(
                    request_tokens=getattr(chunk.usage, 'prompt_tokens', None),
                    response_tokens=getattr(chunk.usage, 'completion_tokens', None),
                    total_tokens=getattr(chunk.usage, 'total_tokens', None),
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            
            # Handle text content
            if choice.delta and choice.delta.content:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=choice.delta.content,
                )
                if maybe_event is not None:
                    yield maybe_event

            # Handle tool calls
            if choice.delta and choice.delta.tool_calls:
                for i, tool_call_delta in enumerate(choice.delta.tool_calls):
                    if tool_call_delta.function:
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=i,
                            tool_name=tool_call_delta.function.name,
                            args=tool_call_delta.function.arguments,
                            tool_call_id=tool_call_delta.id,
                        )
                        if maybe_event is not None:
                            yield maybe_event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
