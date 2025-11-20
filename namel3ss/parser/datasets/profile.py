"""Dataset profile parsing for dataset parser."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetProfile


class ProfileParserMixin:
    """Mixin for parsing dataset profiles."""
    
    def _build_dataset_profile(self, data: Dict[str, Any]) -> "DatasetProfile":
        """Build a DatasetProfile from parsed data."""
        from ...ast import DatasetProfile
        
        if not isinstance(data, dict):
            return DatasetProfile()
        
        profile_data = dict(data)
        row_count = self._coerce_int(profile_data.pop('row_count', profile_data.pop('rows', None)))
        column_count = self._coerce_int(profile_data.pop('column_count', profile_data.pop('columns', None)))
        freshness_raw = profile_data.pop('freshness', profile_data.pop('recency', None))
        freshness = str(freshness_raw) if freshness_raw is not None else None
        updated_raw = profile_data.pop('updated_at', profile_data.pop('last_updated', profile_data.pop('updated', None)))
        updated_at = str(updated_raw) if updated_raw is not None else None
        stats_raw = profile_data.pop('stats', {})
        stats = self._coerce_options_dict(stats_raw)
        
        if profile_data:
            extras_bucket = stats.setdefault('extras', {})
            if not isinstance(extras_bucket, dict):
                extras_bucket = {}
                stats['extras'] = extras_bucket
            extras_bucket.update(profile_data)
        
        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            freshness=freshness,
            updated_at=updated_at,
            stats=stats,
        )

