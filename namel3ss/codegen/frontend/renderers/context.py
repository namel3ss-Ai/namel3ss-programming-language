"""Rendering context passed between statement renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import App, Page

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from ..preview import PreviewDataResolver


@dataclass
class RenderContext:
    app: App
    page: Page
    slug: str
    backend_slug: str
    body_lines: List[str]
    inline_scripts: List[str]
    counters: Dict[str, int]
    widget_defs: List[Dict[str, Any]]
    theme_mode: Optional[str]
    component_tracker: Dict[str, int]
    preview: "PreviewDataResolver"
    scope: str = "page"

    def nested(self) -> "RenderContext":
        """Return a nested rendering context for control-flow blocks."""
        return RenderContext(
            app=self.app,
            page=self.page,
            slug=self.slug,
            backend_slug=self.backend_slug,
            body_lines=self.body_lines,
            inline_scripts=self.inline_scripts,
            counters=self.counters,
            widget_defs=self.widget_defs,
            theme_mode=self.theme_mode,
            component_tracker=self.component_tracker,
            preview=self.preview,
            scope="nested",
        )
