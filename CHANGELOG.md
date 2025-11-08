## `0.2.2` - Jan 8, 2025

- Upgrades to latest stable versions of dependencies:
  - `litellm` from `1.76.3` to `1.79.1`
  - `pydantic-ai-slim` from `1.0.1` to `1.12.0`
- All tests pass and functionality verified with updated dependencies
- Maintains backward compatibility

## `0.2.1` - Aug 31, 2025

- Bug fixes and improvements

## `0.2.0` - Aug 31, 2025

`0.1.0` was broken with newer version of `pydantic-ai-slim`.

- Upgrades to `pydantic-ai-slim` version `0.8.1`
- Changes in `LiteLLMModel` to be compatible with the updated `pydantic-ai-slim` version
- Exposes `LiteLLMModelSettings` which was previously unexpectedly not exposed

## `0.1.0` - Aug 13, 2025

- First version of `pydantic-ai-litellm`
- Uses `pydantic-ai-slim` version `0.6.2`
