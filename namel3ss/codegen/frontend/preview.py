"""Generate deterministic preview data for frontend widgets."""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from namel3ss.ast import (
    AggregateOp,
    App,
    ComputedColumnOp,
    Dataset,
    DatasetSchemaField,
    GroupByOp,
    ShowChart,
    ShowTable,
)

_CATEGORY_OVERRIDES: Dict[str, Sequence[str]] = {
    "status": ("Pending", "In Progress", "Completed", "Archived"),
    "state": ("Draft", "Active", "Suspended", "Closed"),
    "stage": ("Prospect", "Qualified", "Negotiation", "Won"),
    "region": ("North America", "Europe", "APAC", "LATAM"),
    "country": ("USA", "Germany", "Japan", "Brazil"),
    "segment": ("Enterprise", "Midmarket", "SMB", "Consumer"),
    "priority": ("Low", "Medium", "High", "Urgent"),
    "channel": ("Online", "Retail", "Wholesale", "Field"),
}

_MONTH_NAMES: Tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_BOOLEAN_WORDS = {"yes", "no", "true", "false"}


@dataclass
class PreviewTable:
    """Container for preview rows and metadata."""

    name: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    schema: Dict[str, DatasetSchemaField]
    source_label: str


class PreviewDataResolver:
    """Creates deterministic preview records for tables and charts."""

    def __init__(self, app: App, sample_size: int = 6) -> None:
        self._app = app
        self._sample_size = max(3, sample_size)
        self._dataset_lookup: Dict[str, Dataset] = {dataset.name: dataset for dataset in app.datasets}
        self._dataset_previews: Dict[str, PreviewTable] = {}
        self._table_previews: Dict[str, PreviewTable] = {}
        self._build_previews()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def table_preview(self, stmt: ShowTable) -> Dict[str, Any]:
        preview = self._resolve_source_preview(stmt.source_type, stmt.source)
        requested_columns = list(stmt.columns) if stmt.columns else None
        seed = f"table:{stmt.source}:{stmt.title}"

        if preview:
            columns = requested_columns or preview.columns
            projected_rows = self._project_rows(preview, columns, seed)
        else:
            columns = requested_columns or self._fallback_columns(stmt.source)
            projected_rows = [self._build_fallback_row(columns, seed, index) for index in range(self._sample_size)]

        filtered_rows = self._apply_filter(projected_rows, stmt.filter_by) if stmt.filter_by else projected_rows
        if not filtered_rows:
            filtered_rows = [self._build_fallback_row(columns, seed + "#filtered", index) for index in range(self._sample_size)]

        if stmt.sort_by:
            filtered_rows = self._apply_sort(filtered_rows, stmt.sort_by)

        return {
            "columns": columns,
            "rows": filtered_rows[: self._sample_size],
        }

    def chart_preview(self, stmt: ShowChart) -> Dict[str, Any]:
        preview = self._resolve_source_preview(stmt.source_type, stmt.source)
        seed = f"chart:{stmt.source}:{stmt.heading}"
        if preview:
            base_columns = preview.columns
            base_rows = preview.rows[: self._sample_size]
        else:
            base_columns = []
            base_rows = []

        if not base_rows:
            fallback_columns = [stmt.x, stmt.y] if stmt.x and stmt.y else None
            inferred_columns = [col for col in (fallback_columns or []) if col]
            if not inferred_columns:
                inferred_columns = ["category", "value"]
            base_rows = [self._build_fallback_row(inferred_columns, seed, index) for index in range(self._sample_size)]
            base_columns = inferred_columns

        x_key = stmt.x or self._resolve_default_dimension(base_columns)
        metric_candidates = self._resolve_metric_candidates(stmt, base_rows, base_columns, exclude={x_key})
        datasets = []
        for idx, metric in enumerate(metric_candidates):
            data_points = []
            for row_index, row in enumerate(base_rows):
                value = row.get(metric)
                numeric = self._coerce_number(value)
                if numeric is None:
                    numeric = self._fallback_numeric_value(metric, seed, row_index)
                data_points.append(numeric)
            dataset_label = self._prettify_label(metric) or (stmt.heading if idx == 0 else f"Series {idx + 1}")
            datasets.append({
                "label": dataset_label,
                "data": data_points,
            })

        labels = []
        for row_index, row in enumerate(base_rows):
            raw_label = row.get(x_key)
            if raw_label is None:
                raw_label = self._fallback_category_value(x_key, seed, row_index)
            labels.append(str(raw_label))

        return {
            "labels": labels,
            "datasets": datasets or [
                {
                    "label": stmt.heading or "Series",
                    "data": [self._fallback_numeric_value("value", seed, index) for index in range(len(labels))],
                }
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_previews(self) -> None:
        for dataset in self._dataset_lookup.values():
            preview = self._build_dataset_preview(dataset)
            if not preview:
                continue
            self._dataset_previews[dataset.name] = preview
            # Reuse table previews when dataset sources a table directly
            if dataset.source_type == "table" and dataset.source:
                self._table_previews.setdefault(dataset.source, preview)

    def _resolve_source_preview(self, source_type: str, source_name: str) -> Optional[PreviewTable]:
        source_type = (source_type or "").lower()
        if source_type == "dataset" and source_name in self._dataset_previews:
            return self._dataset_previews[source_name]
        if source_type == "table" and source_name in self._table_previews:
            return self._table_previews[source_name]
        return None

    def _build_dataset_preview(self, dataset: Dataset) -> Optional[PreviewTable]:
        columns = self._infer_dataset_columns(dataset)
        if not columns:
            return None

        schema_map = {field.name: field for field in dataset.schema if field.name}
        seed_prefix = f"dataset:{dataset.name}"
        row_count_hint = getattr(dataset.profile, "row_count", None) if dataset.profile else None
        total_rows = min(self._sample_size, row_count_hint or self._sample_size)

        rows = [
            self._build_row_from_schema(columns, schema_map, seed_prefix, index)
            for index in range(total_rows)
        ]
        return PreviewTable(
            name=dataset.name,
            columns=columns,
            rows=rows,
            schema=schema_map,
            source_label=dataset.source or dataset.name,
        )

    def _infer_dataset_columns(self, dataset: Dataset) -> List[str]:
        if dataset.schema:
            return [field.name for field in dataset.schema if field.name]

        seen = set()
        columns: List[str] = []

        for op in dataset.operations:
            if isinstance(op, GroupByOp):
                for col in op.columns:
                    canonical = col.strip()
                    if canonical and canonical not in seen:
                        columns.append(canonical)
                        seen.add(canonical)
            elif isinstance(op, AggregateOp):
                alias = self._extract_alias(op.expression)
                if alias and alias not in seen:
                    columns.append(alias)
                    seen.add(alias)
            elif isinstance(op, ComputedColumnOp):
                if op.name and op.name not in seen:
                    columns.append(op.name)
                    seen.add(op.name)

        if not columns:
            # As a last resort, synthesize placeholder columns
            columns = ["id", "value", "status"]
        return columns

    def _extract_alias(self, expression: str) -> Optional[str]:
        if not expression:
            return None
        match = re.search(r"\bas\s+([A-Za-z_][\w]*)", expression)
        if match:
            return match.group(1)
        stripped = expression.strip()
        stripped = stripped.strip('"')
        return stripped or None

    def _build_row_from_schema(
        self,
        columns: Sequence[str],
        schema: Dict[str, DatasetSchemaField],
        seed_prefix: str,
        index: int,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for column in columns:
            field = schema.get(column)
            dtype = field.dtype if field else None
            row[column] = self._synth_value(dtype, column, seed_prefix, index)
        return row

    def _project_rows(self, preview: PreviewTable, columns: Sequence[str], seed: str) -> List[Dict[str, Any]]:
        projected: List[Dict[str, Any]] = []
        schema = preview.schema
        for index, row in enumerate(preview.rows[: self._sample_size]):
            projected_row: Dict[str, Any] = {}
            for column in columns:
                if column in row:
                    projected_row[column] = row[column]
                else:
                    dtype = schema.get(column).dtype if column in schema else None
                    projected_row[column] = self._synth_value(dtype, column, seed, index)
            projected.append(projected_row)
        if not projected:
            projected = [self._build_fallback_row(columns, seed, idx) for idx in range(self._sample_size)]
        return projected

    def _apply_filter(self, rows: List[Dict[str, Any]], expression: str) -> List[Dict[str, Any]]:
        expr = expression.strip()
        match = re.match(r"([A-Za-z_][\w]*)\s*==\s*['\"]([^'\"]+)['\"]", expr)
        if not match:
            return rows
        column, value = match.group(1), match.group(2)
        filtered = [row for row in rows if str(row.get(column, "")) == value]
        return filtered

    def _apply_sort(self, rows: List[Dict[str, Any]], expression: str) -> List[Dict[str, Any]]:
        columns = [part.strip() for part in expression.split(',') if part.strip()]
        sorted_rows = rows
        for column_expr in reversed(columns):
            direction = 1
            if column_expr.lower().endswith(' desc'):
                column = column_expr[:-5].strip()
                direction = -1
            elif column_expr.lower().endswith(' asc'):
                column = column_expr[:-4].strip()
            else:
                column = column_expr
            if not column:
                continue

            def _key(row: Dict[str, Any], col: str = column) -> Any:
                value = row.get(col)
                number = self._coerce_number(value)
                return number if number is not None else str(value)

            sorted_rows = sorted(sorted_rows, key=_key, reverse=(direction < 0))
        return sorted_rows

    def _resolve_default_dimension(self, columns: Sequence[str]) -> str:
        if not columns:
            return "category"
        for candidate in columns:
            dtype = self._infer_dtype_from_name(candidate)
            if dtype not in {"currency", "number", "numeric", "integer", "float", "percentage"}:
                return candidate
        return columns[0]

    def _resolve_metric_candidates(
        self,
        stmt: ShowChart,
        rows: Sequence[Dict[str, Any]],
        columns: Sequence[str],
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> List[str]:
        exclude_set = {item for item in (exclude or []) if item}
        if stmt.y:
            return [stmt.y]
        numeric_columns: List[str] = []
        for column in columns:
            if column in exclude_set:
                continue
            if not rows:
                inferred = self._infer_dtype_from_name(column)
                if inferred in {"integer", "currency", "float", "number", "numeric", "percentage", "score"}:
                    numeric_columns.append(column)
                    continue
            values = [row.get(column) for row in rows]
            if values and all(self._coerce_number(value) is not None for value in values):
                numeric_columns.append(column)
        if numeric_columns:
            return numeric_columns
        return [column for column in columns if column not in exclude_set][:1]

    def _fallback_columns(self, source: str) -> List[str]:
        source_slug = (source or "data").replace('/', '_')
        base = ["id", "name", "status"]
        if "sales" in source_slug.lower() or "revenue" in source_slug.lower():
            base = ["month", "revenue", "region"]
        return base

    def _build_fallback_row(self, columns: Sequence[str], seed: str, index: int) -> Dict[str, Any]:
        schema: Dict[str, DatasetSchemaField] = {}
        row: Dict[str, Any] = {}
        for column in columns:
            row[column] = self._synth_value(None, column, seed, index)
        return row

    def _synth_value(self, dtype: Optional[str], column: str, seed_prefix: str, index: int) -> Any:
        dtype_normalised = (dtype or "").strip().lower()
        inferred = dtype_normalised or self._infer_dtype_from_name(column)
        seed = f"{seed_prefix}:{column}:{index}"

        if inferred in {"int", "integer", "bigint", "smallint", "count", "number", "numeric"}:
            return self._stable_int(seed)
        if inferred in {"float", "double", "currency", "money", "decimal"}:
            return self._stable_float(seed)
        if inferred in {"percentage", "percent", "ratio"}:
            return self._stable_percentage(seed)
        if inferred in {"boolean", "bool"}:
            return self._stable_bool(seed)
        if inferred in {"date", "day", "month"}:
            return self._stable_date(seed, kind=inferred)
        if inferred in {"datetime", "timestamp"}:
            return self._stable_datetime(seed)
        if inferred in {"email"}:
            return self._stable_email(column, seed, index)
        if inferred in {"name"}:
            return self._stable_name(column, seed, index)
        if inferred in {"status", "state", "stage", "region", "segment", "priority", "channel"}:
            return self._stable_category(inferred, seed, index)
        if inferred in {"score"}:
            return self._stable_score(seed)
        return self._stable_text(column, seed, index)

    # ------------------------------------------------------------------
    # Value synthesizers
    # ------------------------------------------------------------------
    def _stable_hash(self, key: str) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _stable_int(self, seed: str) -> int:
        return (self._stable_hash(seed) % 9000) + 100

    def _stable_float(self, seed: str) -> float:
        return round((self._stable_hash(seed) % 900000) / 100.0 + 10.0, 2)

    def _stable_percentage(self, seed: str) -> float:
        return round((self._stable_hash(seed) % 10000) / 100.0, 2)

    def _stable_bool(self, seed: str) -> bool:
        return bool(self._stable_hash(seed) % 2)

    def _stable_date(self, seed: str, *, kind: str) -> str:
        base = _dt.date(2024, 1, 1)
        offset = self._stable_hash(seed) % 120
        value = base + _dt.timedelta(days=offset)
        if kind == "month":
            return value.strftime("%b")
        return value.isoformat()

    def _stable_datetime(self, seed: str) -> str:
        base = _dt.datetime(2024, 1, 1, 9, 0)
        offset_minutes = self._stable_hash(seed) % 720
        value = base + _dt.timedelta(minutes=offset_minutes)
        return value.isoformat(timespec="seconds")

    def _stable_email(self, column: str, seed: str, index: int) -> str:
        domain = f"{column.replace('_', '')}.example.com".lower()
        return f"user{index + 1}@{domain}"

    def _stable_name(self, column: str, seed: str, index: int) -> str:
        base = column.replace('_', ' ').title() or "Name"
        return f"{base} {index + 1}"

    def _stable_category(self, inferred: str, seed: str, index: int) -> str:
        key = inferred.lower()
        options = _CATEGORY_OVERRIDES.get(key)
        if not options and key in {"status", "state", "stage"}:
            options = _CATEGORY_OVERRIDES["status"]
        if not options:
            options = ("Alpha", "Beta", "Gamma", "Delta")
        idx = self._stable_hash(f"{seed}:{index}") % len(options)
        return options[idx]

    def _stable_score(self, seed: str) -> float:
        return round((self._stable_hash(seed) % 1000) / 10.0, 1)

    def _stable_text(self, column: str, seed: str, index: int) -> str:
        name = column.replace('_', ' ').title() or "Value"
        return f"{name} {index + 1}"

    def _fallback_numeric_value(self, column: str, seed: str, index: int) -> float:
        return self._stable_float(f"fallback:{column}:{seed}:{index}")

    def _fallback_category_value(self, column: str, seed: str, index: int) -> str:
        inferred = self._infer_dtype_from_name(column)
        if inferred in {"status", "state", "stage", "region", "segment"}:
            return self._stable_category(inferred, seed, index)
        if inferred in {"month"}:
            return _MONTH_NAMES[(index + self._stable_hash(seed)) % len(_MONTH_NAMES)]
        if inferred in {"name"}:
            return self._stable_name(column, seed, index)
        return self._stable_text(column, seed, index)

    def _coerce_number(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            lowered = stripped.lower()
            if lowered in _BOOLEAN_WORDS:
                return 1.0 if lowered in {"yes", "true"} else 0.0
            stripped = stripped.replace('%', '').replace(',', '')
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    def _infer_dtype_from_name(self, column: str) -> str:
        name = (column or "").lower()
        if not name:
            return "string"
        if name.endswith('_id') or name == 'id':
            return "integer"
        if 'email' in name:
            return "email"
        if any(token in name for token in ('amount', 'revenue', 'price', 'cost', 'total', 'balance')):
            return "currency"
        if any(token in name for token in ('pct', 'percent', 'ratio', 'rate')):
            return "percentage"
        if any(token in name for token in ('score', 'rating', 'metric')):
            return "score"
        if any(token in name for token in ('date', 'day')):
            return "date"
        if any(token in name for token in ('time', 'timestamp', 'at')):
            return "datetime"
        if 'month' in name:
            return "month"
        if any(token in name for token in ('status', 'state', 'stage')):
            return "status"
        for marker, dtype_name in (
            ("region", "region"),
            ("country", "region"),
            ("segment", "segment"),
            ("channel", "channel"),
            ("priority", "priority"),
        ):
            if marker in name:
                return dtype_name
        if 'name' in name or 'title' in name:
            return "name"
        return "string"

    def _prettify_label(self, name: str) -> str:
        if not name:
            return ""
        parts = name.replace('_', ' ').split()
        return ' '.join(part.capitalize() for part in parts)
