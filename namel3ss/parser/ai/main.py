"""Main AIParserMixin composition combining all AI parsing capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chains import ChainsParserMixin
from .models import ModelsParserMixin
from .prompts import PromptsParserMixin
from .rlhf import RLHFParserMixin
from .schemas import SchemaParserMixin
from .training import TrainingParserMixin
from .workflows import WorkflowParserMixin

if TYPE_CHECKING:
    from ..base import ParserBase


class AIParserMixin(
    ModelsParserMixin,
    ChainsParserMixin,
    PromptsParserMixin,
    RLHFParserMixin,
    SchemaParserMixin,
    TrainingParserMixin,
    WorkflowParserMixin,
):
    """Comprehensive AI parsing mixin combining all AI-related parsing capabilities."""
    pass
