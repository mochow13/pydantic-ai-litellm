# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pydantic-ai-litellm` is a thin adapter that implements Pydantic AI's `Model` interface on top of LiteLLM, so any of LiteLLM's 100+ providers can be used as a Pydantic AI model. There are two model classes, each in its own file, re-exported from `pydantic_ai_litellm/__init__.py`:

- `LiteLLMModel` / `LiteLLMModelSettings` (`pydantic_ai_litellm/litellm_model.py`) — Chat Completions, via `litellm.acompletion`. The default choice; works with all LiteLLM providers.
- `LiteLLMResponsesModel` / `LiteLLMResponsesModelSettings` (`pydantic_ai_litellm/responses_model.py`) — Responses API, via `litellm.aresponses`. Needed for reasoning-capable OpenAI models (e.g. `gpt-5.x`) that reject function tools on `/v1/chat/completions` and require `/v1/responses` instead.

The two APIs have incompatible request/response shapes (`messages`/`choices` vs. `input`/`output` items, different streaming event types), which is why this is two parallel implementations rather than a branch inside one — see `responses_model.py`'s module docstring for the reasoning. Both mirror each other's structure closely (same method names/shapes: `_map_messages`, `_process_response`, `_get_tools`, `_map_tool_definition`, a `*StreamedResponse` dataclass with the same four properties); when fixing a bug or changing behavior in one, check whether the same fix applies to the other.

Because it's a compatibility shim against two fast-moving upstream libraries (`pydantic-ai-slim`, `litellm`), most historical bugs (see `CHANGELOG.md`) have come from upstream API changes — abstract methods added to `StreamedResponse`, `handle_text_delta` changing its return type, `_get_instructions` being renamed to `_get_instruction_parts`, etc. When something breaks after a dependency bump, check the upstream changelog/source for the method being overridden or called before assuming a logic bug.

## Commands

```bash
# Install deps (also installs the package itself in editable mode via the hatchling build backend)
uv sync

# Run unit tests (this is what CI runs)
uv run pytest tests/unit_tests/ -v

# Run a single test file / test
uv run pytest tests/unit_tests/test_streaming.py -v
uv run pytest tests/unit_tests/test_streaming.py::TestStreaming::test_streamed_text_response_yields_text_events -v

# Integration tests (NOT run in CI; require live provider credentials in a .env file:
# LITELLM_API_KEY, LITELLM_API_BASE, DEEPSEEK_MODEL, ANTHROPIC_MODEL, OPENAI_MODEL)
uv run pytest tests/integration_tests/ -v

# Build distributions (as done by the release workflow)
uv build
```

There is no lint/format/type-check tooling configured in this repo (no ruff/mypy config in `pyproject.toml`).

## Architecture

`LiteLLMModel` (in `litellm_model.py`) implements Pydantic AI's `Model` abstract base class:

- `request()` / `request_stream()` are the two entry points called by the Pydantic AI `Agent`. Both delegate to `_completion_create`, which builds the `litellm.acompletion(**kwargs)` call: maps `ModelSettings` fields to LiteLLM kwargs, and additionally supports `litellm_*`-prefixed settings (`litellm_api_key`, `litellm_api_base`, `litellm_custom_llm_provider`, `litellm_metadata`) defined in `LiteLLMModelSettings`. These `litellm_*` settings fall back to the constructor args (`api_key`, `api_base`, `custom_llm_provider`) if not set per-request. **Convention: any new `LiteLLMModelSettings` field must be `litellm_`-prefixed** so it doesn't collide when settings are merged with other Pydantic AI model settings.
- `_map_messages` converts Pydantic AI's `ModelMessage` list (`ModelRequest`/`ModelResponse` with typed parts like `SystemPromptPart`, `UserPromptPart`, `ToolReturnPart`, `RetryPromptPart`, `TextPart`, `ToolCallPart`) into OpenAI-style `dict` messages that LiteLLM expects. Agent instruction parts (`model_request_parameters.instruction_parts`) are inserted as system messages before the first non-system message, then `_merge_leading_system_messages` collapses consecutive leading system messages into one (required because some strict OpenAI-compatible backends reject multiple leading system messages).
- `_process_response` converts a non-streamed LiteLLM response back into a Pydantic AI `ModelResponse`. It must NOT set `usage.requests` — that's counted by the Pydantic AI framework itself (regression covered by `test_process_response_does_not_set_requests`).
- `LiteLLMStreamedResponse` (a `StreamedResponse` subclass) drives streaming via `_get_event_iterator`, feeding chunks through Pydantic AI's `ModelPartsManager` (`self._parts_manager.handle_text_delta` / `handle_tool_call_delta`). Note the asymmetry: `handle_text_delta` returns an `Iterator[ModelResponseStreamEvent]` (must be iterated/yielded from, e.g. `for event in ...: yield event`), while `handle_tool_call_delta` returns `ModelResponseStreamEvent | None` (must be null-checked). Conflating these two shapes was a real historical bug (see `test_streaming.py` docstring and the `0.2.8` changelog entry) — don't "simplify" one to match the other.
- All content is currently mapped as plain strings; multimodal `UserPromptPart` content (images, files, etc.) is flattened to a joined string rather than structured content parts — a known limitation, not an oversight, if you touch `_map_messages`.

`LiteLLMResponsesModel` (in `responses_model.py`) mirrors the above but for `/v1/responses`:

- Its `_map_messages` produces flat `input` items instead: `{'type': 'message', 'role': ..., 'content': ...}` for text, `{'type': 'function_call', 'call_id', 'name', 'arguments'}` for a tool call, `{'type': 'function_call_output', 'call_id', 'output'}` for a tool result — no nested `tool_calls` list the way Chat Completions nests them under one assistant message.
- Agent instructions go through a dedicated top-level `instructions` string (built by `_build_instructions`) rather than being merged into leading messages — the Responses API has a native field for this, so there's no `_merge_leading_system_messages`-style hack here.
- Response items (`response.output`) and streaming events may come back as either plain dicts or typed objects depending on the code path; both `_process_response` and `LiteLLMResponsesStreamedResponse._get_event_iterator` read them through the module-level `_get_field(obj, name, default)` helper rather than direct attribute/`.get()` access — keep using it for any new field reads here.
- Settings mapping differs from Chat Completions in a few spots: `max_tokens` → `max_output_tokens` (the Responses API's name for it), and `stop_sequences`/`seed` are intentionally *not* forwarded (the Responses API has no equivalent parameter for either).

## Testing conventions

- `tests/unit_tests/` — fast, no network, mock `litellm.acompletion` (`@patch('pydantic_ai_litellm.litellm_model.acompletion')`) or `litellm.aresponses` (`@patch('pydantic_ai_litellm.responses_model.aresponses')`), or fabricate response/chunk objects with `unittest.mock.Mock`. This is what CI runs; keep new tests here unless they genuinely need a live provider. Tests for the two models are split into parallel file pairs (`test_map_messages.py` / `test_responses_map_messages.py`, `test_streaming.py` / `test_responses_streaming.py`, `test_tool_calling.py` / `test_responses_tool_calling.py`) — keep that pairing when adding coverage for one model or the other.
- `tests/integration_tests/` — exercise real providers through LiteLLM, driven by env vars loaded from `.env` via `python-dotenv`. Not run in CI; run manually when credentials are available.
- Async tests use `@pytest.mark.asyncio` explicitly (no global `asyncio_mode = auto` configured), so new async tests need the marker.
- When a test is a regression test for a specific upstream breakage, mirror the existing style in `test_streaming.py`: a module/class docstring explaining *what broke upstream and why*, not just what the test checks.

## Release process

Versioning lives only in `pyproject.toml` (`[project].version`) — there's no `__version__` string to keep in sync, since `pydantic_ai_litellm/__init__.py` derives `__version__` from installed package metadata at runtime. Publishing to PyPI (`.github/workflows/pypi-release.yml`) is triggered by creating a GitHub Release; it builds with `uv build` and uses PyPI trusted publishing. `CHANGELOG.md` is maintained by hand — add an entry per release, and note when a change affects the dependency floors on `litellm`/`pydantic-ai-slim` or is a breaking change for subclassers of `LiteLLMModel`.
