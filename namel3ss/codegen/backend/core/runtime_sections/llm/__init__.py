"""LLM runtime section package - assembles all LLM code generation components."""

from .imports import IMPORTS
from .utilities import UTILITIES
from .http_client import HTTP_CLIENT
from .response_parser import RESPONSE_PARSER
from .cache import LLM_CACHE
from .metrics import LLM_METRICS
from .batching import LLM_BATCHING
from .circuit_breaker import CIRCUIT_BREAKER
from .tools import TOOLS
from .workflow import WORKFLOW
from .prompts import PROMPTS
from .connectors import CONNECTORS
from .structured import STRUCTURED
from .main import MAIN

# Compose the complete LLM_SECTION from all modules
LLM_SECTION = '\n\n\n'.join([
    IMPORTS,
    UTILITIES,
    HTTP_CLIENT,
    RESPONSE_PARSER,
    LLM_CACHE,
    LLM_METRICS,
    LLM_BATCHING,
    CIRCUIT_BREAKER,
    TOOLS,
    WORKFLOW,
    PROMPTS,
    CONNECTORS,
    STRUCTURED,
    MAIN,
])

__all__ = ['LLM_SECTION']
