"""Unified error model for Namel3ss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorLocation:
    path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

    def describe(self) -> str:
        if self.path and self.line is not None and self.column is not None:
            return f"{self.path}:{self.line}:{self.column}"
        if self.path and self.line is not None:
            return f"{self.path}:{self.line}"
        if self.path:
            return self.path
        return "unknown location"


class N3Error(Exception):
    """Base class for all compiler/runtime errors surfaced to users."""

    code: Optional[str] = None
    hint: Optional[str] = None

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        code: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.location = ErrorLocation(path=path, line=line, column=column)
        self.path = path
        self.line = line
        self.column = column
        if code is not None:
            self.code = code
        if hint is not None:
            self.hint = hint

    def format(self) -> str:
        components = [self.message]
        meta_parts = []
        location_desc = self.location.describe()
        if location_desc != "unknown location":
            meta_parts.append(location_desc)
        if self.code:
            meta_parts.append(self.code)
        if meta_parts:
            components[-1] = f"{components[-1]} ({'; '.join(meta_parts)})"
        if self.hint:
            components.append(f"Hint: {self.hint}")
        return " ".join(part for part in components if part)


class N3SyntaxError(N3Error):
    """Raised when the parser encounters invalid syntax."""


class N3TypeError(N3Error):
    """Raised when static type checking fails."""


class N3ResolutionError(N3Error):
    """Raised when module or import resolution fails."""


class N3FrameOpError(N3Error):
    """Raised when frame/dataset operations are invalid."""


class N3EffectError(N3Error):
    """Raised when effect analysis fails."""


class N3EvaluationError(N3Error):
    """Raised when evaluation/runtime execution fails."""


__all__ = [
    "N3Error",
    "N3SyntaxError",
    "N3TypeError",
    "N3ResolutionError",
    "N3FrameOpError",
    "N3EffectError",
    "N3EvaluationError",
    "ErrorLocation",
]
