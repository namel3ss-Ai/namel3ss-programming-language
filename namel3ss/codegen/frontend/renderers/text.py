"""Rendering helpers for text statements."""

from __future__ import annotations

import html
from typing import Optional

from namel3ss.ast import ShowText

from ..placeholders import TEXT_PLACEHOLDER_RE
from ..theme import style_to_inline
from .context import RenderContext


def render_text(stmt: ShowText, ctx: RenderContext, component_index: Optional[int]) -> None:
    styles = style_to_inline(stmt.styles)
    template = stmt.text
    initial_text = TEXT_PLACEHOLDER_RE.sub("", template)
    attrs = [f'data-n3-text-template="{html.escape(template, quote=True)}"']
    if styles:
        attrs.append(f'style="{styles}"')
    attr_str = " ".join(attrs)
    ctx.body_lines.append(f"<p {attr_str}>{html.escape(initial_text)}</p>")
