## `0.2.8` - Jun 2, 2026

- Fix streaming for `pydantic-ai-slim` 1.103: implement `provider_url` on `LiteLLMStreamedResponse` (previously an unimplemented abstract method) and consume `handle_text_delta` as an iterator instead of a single event ([#13](https://github.com/mochow13/pydantic-ai-litellm/pull/13)).

## `0.2.7` - Jun 1, 2026

- Fix double-counting of `RunUsage.requests` in non-streamed requests ([#11](https://github.com/mochow13/pydantic-ai-litellm/pull/11)).
- Add `[build-system]` table (`hatchling`) to `pyproject.toml` so `uv sync` correctly installs the package in editable mode.
- CI: add GitHub Actions workflow that runs unit tests across Python 3.10–3.13 on every PR and merge to `main`.

## `0.2.6` - May 28, 2026

- Fix compatibility with `pydantic-ai-slim` >=1.95: replace removed `_get_instructions` with `_get_instruction_parts` ([#8](https://github.com/mochow13/pydantic-ai-litellm/issues/8)).
- Merge consecutive leading system messages for strict LiteLLM/vLLM backends.
- Dependencies: require `litellm>=1.86.2` and `pydantic-ai-slim>=1.95.0` (raised from `>=1.83.8` / `>=1.82.0` in `0.2.5`).

**Notes for upgraders** (public `LiteLLMModel` / `LiteLLMModelSettings` API is unchanged):

- **Dependency floors**: upgrading from `0.2.5` requires `pydantic-ai-slim` >=1.95 and `litellm` >=1.86.2; older pins will not resolve.
- **Subclasses**: `_map_messages` now takes `model_request_parameters`; override it only with the updated signature.
- **Instruction message ordering**: agent instructions are inserted before the first non-system message and consecutive leading system messages are merged with `\n\n` — a subtle behavior change vs always prepending a single system message at index 0.

## `0.2.5` - Apr 16, 2026

- New upload on PyPI: `0.2.4` artifacts cannot be overwritten ([file name reuse](https://pypi.org/help/#file-name-reuse)), so this patch increments the version only for publishing.
- README: add shields.io-style badges (version, Python versions, license, downloads, GitHub).
- Examples: add `examples/07_install_from_pypi.py` (install-from-PyPI flow with `OPENAI_API_KEY` and optional `MODEL_NAME`) and document it in `examples/README.md`.

## `0.2.4` - Apr 16, 2026

- Dependencies: require `litellm>=1.83.8` and `pydantic-ai-slim>=1.82.0`.
- Regenerate `uv.lock` for the new lower bounds.

## `0.2.3` - Nov 19, 2025

- **Python**: lower `requires-python` from `>=3.13` to `>=3.10`; extend PyPI classifiers through 3.13.
- **Tooling**: update `.python-version` for local development.
- **Dependencies**: keep `litellm>=1.79.1` and `pydantic-ai-slim>=1.12.0` (same floors as `0.2.2`).

## `0.2.2` - Nov 8, 2025

- **Dependencies**: raise minimum `litellm` to `>=1.79.1` and `pydantic-ai-slim` to `>=1.12.0` (replacing earlier floors such as `1.76.3` / `1.0.1`).
- **Python**: package metadata still required **3.13+** until relaxed in `0.2.3`.
- Tests and integration verified against the upgraded stack.

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
