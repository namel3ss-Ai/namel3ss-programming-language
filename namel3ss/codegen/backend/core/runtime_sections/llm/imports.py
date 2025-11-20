"""Imports section for LLM runtime code generation."""

from textwrap import dedent

IMPORTS = dedent(
    '''
import asyncio
import copy
import inspect
import json
import os
import time
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from namel3ss.codegen.backend.core.runtime.expression_sandbox import (
    evaluate_expression_tree as _evaluate_expression_tree,
)

# Structured prompt support
try:
    from namel3ss.prompts import PromptProgram, execute_structured_prompt_sync
    from namel3ss.llm.base import BaseLLM
    from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
    _STRUCTURED_PROMPTS_AVAILABLE = True
except ImportError:
    _STRUCTURED_PROMPTS_AVAILABLE = False
'''
).strip()

__all__ = ['IMPORTS']
