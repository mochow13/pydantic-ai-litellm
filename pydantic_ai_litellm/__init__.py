from importlib import metadata

from .litellm_model import LiteLLMModel, LiteLLMModelSettings
from .responses_model import LiteLLMResponsesModel, LiteLLMResponsesModelSettings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "LiteLLMModel",
    "LiteLLMModelSettings",
    "LiteLLMResponsesModel",
    "LiteLLMResponsesModelSettings",
    "__version__",
]
