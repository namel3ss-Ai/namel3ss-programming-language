"""Utility builders for policies and configurations."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ...ast import CachePolicy, PaginationPolicy, StreamingPolicy


class UtilitiesMixin:
    """Mixin providing utility methods for building policies."""
    
    def _default_app_name(self) -> str:
        """Generate default app name from module or file name."""
        if self.module_name:
            return self.module_name.split('.')[-1]
        if self.source_path:
            stem = Path(self.source_path).stem
            if stem:
                return stem
        return "app"
    
    def _build_cache_policy(self, data: Dict[str, Any]) -> "CachePolicy":
        """Build CachePolicy from configuration data."""
        from namel3ss.ast import CachePolicy
        
        if not data:
            return CachePolicy(strategy="none")
        
        strategy = str(data.get('strategy', 'memory') or 'memory').lower()
        ttl_raw = data.get('ttl_seconds') or data.get('ttl') or data.get('ttl_s')
        ttl_seconds: Optional[int] = None
        
        if ttl_raw is not None:
            if isinstance(ttl_raw, (int, float)):
                ttl_seconds = int(ttl_raw)
            else:
                ttl_clean = str(ttl_raw).strip()
                match_val = re.match(r'(\d+)', ttl_clean)
                if match_val:
                    ttl_seconds = int(match_val.group(1))
        
        max_entries = data.get('max_entries') or data.get('max_rows') or data.get('max')
        if max_entries is not None and not isinstance(max_entries, int):
            try:
                max_entries = int(str(max_entries))
            except ValueError:
                max_entries = None
        
        return CachePolicy(strategy=strategy, ttl_seconds=ttl_seconds, max_entries=max_entries)

    def _build_pagination_policy(self, data: Dict[str, Any]) -> "PaginationPolicy":
        """Build PaginationPolicy from configuration data."""
        from namel3ss.ast import PaginationPolicy
        
        if not data:
            return PaginationPolicy(enabled=False)
        
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        page_size = data.get('page_size') or data.get('page_size') or data.get('size')
        
        if page_size is not None and not isinstance(page_size, int):
            try:
                page_size = int(str(page_size))
            except ValueError:
                page_size = None
        
        max_pages = data.get('max_pages') or data.get('max_pages')
        if max_pages is not None and not isinstance(max_pages, int):
            try:
                max_pages = int(str(max_pages))
            except ValueError:
                max_pages = None
        
        return PaginationPolicy(enabled=enabled, page_size=page_size, max_pages=max_pages)

    def _build_streaming_policy(self, data: Dict[str, Any]) -> "StreamingPolicy":
        """Build StreamingPolicy from configuration data."""
        from namel3ss.ast import StreamingPolicy
        
        if not data:
            return StreamingPolicy(enabled=True)
        
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        chunk_size = data.get('chunk_size') or data.get('chunk_size') or data.get('batch')
        
        if chunk_size is not None and not isinstance(chunk_size, int):
            try:
                chunk_size = int(str(chunk_size))
            except ValueError:
                chunk_size = None
        
        return StreamingPolicy(enabled=enabled, chunk_size=chunk_size)
