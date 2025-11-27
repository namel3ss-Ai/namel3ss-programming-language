"""Utilities for validating and capturing Namel3ss comments."""

from __future__ import annotations

import re
from typing import Optional

from namel3ss.ast.comments import Comment

# Regex equivalents for the documented comment rules
COMMENT_PATTERN = re.compile(r"^\s*#\s\S.*$")
EMOJI_COMMENT_PATTERN = re.compile(
    r"^\s*#\s(?:[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]\s?)?.*$"
)
INLINE_COMMENT_PATTERN = re.compile(r"#\s\S.*$")


def _is_emoji_char(ch: str) -> bool:
    """Lightweight emoji detector using common Unicode ranges."""
    if not ch:
        return False
    return bool(
        re.match(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]", ch)
    )


def comment_error_for_line(line: str) -> Optional[str]:
    """Return a validation error message if the line uses an unsupported comment style."""
    stripped = line.lstrip()
    if not stripped:
        return None
    if stripped.startswith("//"):
        return "Only '#' single-line comments are supported; '//' comments are not allowed"
    if stripped.startswith("/*"):
        return "Block comments are not supported; use '# 💬 comment text' instead"
    if stripped.startswith("#") and not COMMENT_PATTERN.match(stripped):
        return "Comments must start with '# ' followed by text"
    return None


def is_comment_text(text: str) -> bool:
    """Check if the provided text fragment is a valid Namel3ss comment."""
    return COMMENT_PATTERN.match(text.lstrip()) is not None


def parse_comment_metadata(line: str, line_no: int, column_override: Optional[int] = None) -> Comment:
    """Extract structured comment metadata from a line of source."""
    idx = line.find("#")
    if idx == -1:
        raise ValueError("No comment marker found in line")

    fragment = line[idx:]
    reason = comment_error_for_line(fragment)
    if reason:
        raise ValueError(reason)

    raw = fragment.rstrip("\n")
    body = raw[1:].lstrip()
    if not body:
        raise ValueError("Comments must include text after '# '")

    emoji = None
    remaining = body
    first_token = body.split(maxsplit=1)[0]
    if first_token and _is_emoji_char(first_token[0]):
        emoji = first_token[0]
        remaining = body[len(first_token):].lstrip()

    text = remaining or ""
    column = column_override or (idx + 1)
    return Comment(raw=raw, text=text, emoji=emoji, line=line_no, column=column)


def has_emoji_prefix(line: str) -> bool:
    """Check whether a comment starts with an emoji marker."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False
    body = stripped[1:].lstrip()
    if not body:
        return False
    first = body.split(maxsplit=1)[0]
    return _is_emoji_char(first[0])
