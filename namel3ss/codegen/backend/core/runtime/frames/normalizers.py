"""Column and expression normalization utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


def _normalize_select_columns(columns: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, column in enumerate(columns):
        if isinstance(column, str):
            normalized.append(
                {
                    "source": column,
                    "alias": column,
                    "expression": None,
                    "expression_source": None,
                }
            )
            continue
        if isinstance(column, dict):
            alias = column.get("alias") or column.get("name") or column.get("source")
            source = column.get("source") or column.get("name")
            expression = column.get("expression")
            normalized.append(
                {
                    "source": source,
                    "alias": alias or f"column_{idx}",
                    "expression": expression,
                    "expression_source": column.get("expression_source"),
                }
            )
    return normalized


def _normalize_order_columns(columns: Sequence[Any], default_descending: bool) -> List[Dict[str, Any]]:
    order_specs: List[Dict[str, Any]] = []
    for column in columns:
        name: Optional[str] = None
        descending_value: Optional[bool] = None
        if isinstance(column, dict):
            name = column.get("name") or column.get("column")
            if isinstance(column.get("descending"), bool):
                descending_value = bool(column.get("descending"))
            elif "descending" in column:
                descending_value = bool(column.get("descending"))
            elif "desc" in column:
                descending_value = bool(column.get("desc"))
        else:
            name = str(column)
        if not name:
            continue
        if descending_value is None:
            descending_value = bool(default_descending)
        order_specs.append({"name": name, "descending": descending_value})
    return order_specs


def _coerce_expression_spec(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"type": "name", "name": value}
    return {"type": "literal", "value": value}


def _normalize_join_expressions(join_expressions: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, expression in enumerate(join_expressions or []):
        entry: Optional[Dict[str, Any]] = None
        if isinstance(expression, dict):
            left_expr = _coerce_expression_spec(expression.get("left_expression") or expression.get("left"))
            right_expr = _coerce_expression_spec(expression.get("right_expression") or expression.get("right"))
            if left_expr is None or right_expr is None:
                continue
            entry = {
                "name": f"__n3_join_key_{idx}",
                "left_expression": left_expr,
                "right_expression": right_expr,
                "left_expression_source": expression.get("left_expression_source") or expression.get("left_source"),
                "right_expression_source": expression.get("right_expression_source") or expression.get("right_source"),
            }
        elif isinstance(expression, (list, tuple)) and len(expression) == 2:
            left_expr = _coerce_expression_spec(expression[0])
            right_expr = _coerce_expression_spec(expression[1])
            if left_expr is None or right_expr is None:
                continue
            entry = {
                "name": f"__n3_join_key_{idx}",
                "left_expression": left_expr,
                "right_expression": right_expr,
                "left_expression_source": None,
                "right_expression_source": None,
            }
        if entry:
            normalized.append(entry)
    return normalized


def _collect_right_column_names(
    join_rows: Sequence[Dict[str, Any]],
    join_schema: Optional[Sequence[Dict[str, Any]]],
) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for column in join_schema or []:
        name = column.get("name")
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    sample = next((row for row in join_rows if isinstance(row, dict)), None)
    if isinstance(sample, dict):
        for key in sample.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names




__all__ = [
    "_normalize_select_columns",
    "_normalize_order_columns",
    "_coerce_expression_spec",
    "_normalize_join_expressions",
    "_collect_right_column_names",
]
