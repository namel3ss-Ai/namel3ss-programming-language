from __future__ import annotations

from textwrap import dedent

INSIGHTS_SECTION = dedent(
    '''
from namel3ss.codegen.backend.core.runtime.insights import (
    evaluate_insights_for_dataset as _evaluate_insights_for_dataset_impl,
    run_insight as _run_insight_impl,
    evaluate_expression as _evaluate_expression_impl,
    resolve_expression_path as _resolve_expression_path_impl,
)
from namel3ss.codegen.backend.core.sql_compiler import compile_dataset_to_sql


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    return _evaluate_insights_for_dataset_impl(
        name,
        rows,
        context,
        insights=INSIGHTS,
        run_insight=_run_insight,
    )


def _run_insight(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_insight_impl(
        spec,
        rows,
        context,
        model_registry=MODEL_REGISTRY,
        predict_callable=predict,
        evaluate_expression=_evaluate_expression,
        resolve_expression_path=_resolve_expression_path,
        render_template_value=_render_template_value,
    )


def _evaluate_expression(
    expression: Optional[str],
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    return _evaluate_expression_impl(
        expression,
        rows,
        scope,
        resolve_expression_path=_resolve_expression_path,
    )


def _resolve_expression_path(
    expression: str,
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    return _resolve_expression_path_impl(expression, rows, scope)



def evaluate_insight(slug: str, context: Optional[Dict[str, Any]] = None) -> InsightResponse:
    spec = INSIGHTS.get(slug)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Insight '{slug}' is not defined")
    ctx = dict(context or build_context(None))
    rows: List[Dict[str, Any]] = []
    result = evaluate_insights_for_dataset(slug, rows, ctx)
    dataset = result.get("dataset") or spec.get("source_dataset") or slug
    return InsightResponse(name=slug, dataset=dataset, result=result)

    '''
).strip()

__all__ = ['INSIGHTS_SECTION']
