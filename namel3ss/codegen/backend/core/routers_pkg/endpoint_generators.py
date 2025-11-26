from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


def _render_page_endpoint(page: PageSpec) -> List[str]:
    func_name = f"page_{page.slug}_{page.index}"
    path = f"/api/pages/{page.slug}"
    lines = [
        f"@router.get({path!r}, response_model=Dict[str, Any], tags=['pages'])",
        f"async def {func_name}_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:",
        f"    payload = await runtime.{func_name}(session)",
    ]
    if page.reactive:
        lines.append("    if runtime.REALTIME_ENABLED:")
        lines.append(f"        await runtime.broadcast_page_snapshot({page.slug!r}, payload)")
    lines.append("    return payload")
    return lines


def _render_component_endpoint(
    page: PageSpec, component: PageComponent, index: int
) -> List[str]:
    payload = component.payload
    slug = page.slug
    base_path = f"/api/pages/{slug}"
    if component.type == "table":
        insight_name = payload.get("insight")
        source_name = payload.get("source") or ""
        source_type = (payload.get("source_type") or "dataset").lower()
        meta_payload: Dict[str, Any] = {}
        if insight_name:
            meta_payload["insight"] = insight_name
        if source_name:
            meta_payload["source"] = source_name
        meta_expr = _format_literal(meta_payload) if meta_payload else "None"
        return [
            f"@router.get({base_path!r} + '/tables/{index}', response_model=TableResponse, tags=['pages'])",
            f"async def {slug}_table_{index}(session: AsyncSession = Depends(get_session)) -> TableResponse:",
            f"    context = runtime.build_context({page.slug!r})",
            f"    source_type = {source_type!r}",
            f"    source_name = {source_name!r}",
            "    frame_spec = runtime.FRAMES.get(source_name) if source_type == 'frame' else None",
            "    if source_type == 'frame':",
            "        rows = await runtime.fetch_frame_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(frame_spec.get('source')) if frame_spec else None",
            "    else:",
            "        rows = await runtime.fetch_dataset_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(source_name)",
            "    insights: Dict[str, Any] = {}",
            f"    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:",
            f"        if {insight_name!r}:",
            "            try:",
            f"                insights = runtime.evaluate_insights_for_dataset({insight_name!r}, rows, context)",
            "            except Exception:",
            f"                runtime.logger.exception('Failed to evaluate insight %s', {insight_name!r})",
            "                insights = {}",
            "    response = TableResponse(",
            f"        title={payload.get('title')!r},",
            f"        source={{'type': {source_type!r}, 'name': {source_name!r}}},",
            f"        columns={payload.get('columns') or []!r},",
            f"        filter={payload.get('filter')!r},",
            f"        sort={payload.get('sort')!r},",
            f"        style={payload.get('style') or {}!r},",
            f"        insight={insight_name!r},",
            "        rows=rows,",
            "        insights=insights,",
            "    )",
            f"    is_reactive = {page.reactive!r} or (dataset.get('reactive') if dataset else False)",
            "    if runtime.REALTIME_ENABLED and is_reactive:",
            f"        await runtime.broadcast_component_update({page.slug!r}, 'table', {index}, response, meta={meta_expr})",
            "    return response",
        ]
    if component.type == "chart":
        insight_name = payload.get("insight")
        source_name = payload.get("source") or ""
        source_type = (payload.get("source_type") or "dataset").lower()
        meta_payload: Dict[str, Any] = {}
        if payload.get("chart_type"):
            meta_payload["chart_type"] = payload.get("chart_type")
        if insight_name:
            meta_payload["insight"] = insight_name
        if source_name:
            meta_payload["source"] = source_name
        meta_map = {key: value for key, value in meta_payload.items() if value is not None}
        meta_expr = _format_literal(meta_map) if meta_map else "None"
        return [
            f"@router.get({base_path!r} + '/charts/{index}', response_model=ChartResponse, tags=['pages'])",
            f"async def {slug}_chart_{index}(session: AsyncSession = Depends(get_session)) -> ChartResponse:",
            f"    context = runtime.build_context({page.slug!r})",
            f"    source_type = {source_type!r}",
            f"    source_name = {source_name!r}",
            "    frame_spec = runtime.FRAMES.get(source_name) if source_type == 'frame' else None",
            "    if source_type == 'frame':",
            "        rows = await runtime.fetch_frame_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(frame_spec.get('source')) if frame_spec else None",
            "    else:",
            "        rows = await runtime.fetch_dataset_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(source_name)",
            "    labels: List[Any] = [row.get('label', idx) for idx, row in enumerate(rows, start=1)]",
            "    series_values: List[Any] = [row.get('value', idx * 10) for idx, row in enumerate(rows, start=1)]",
            "    series = [{'label': 'Series', 'data': series_values}]",
            "    insight_results: Dict[str, Any] = {}",
            f"    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:",
            f"        if {insight_name!r}:",
            "            try:",
            f"                insight_results = runtime.evaluate_insights_for_dataset({insight_name!r}, rows, context)",
            "            except Exception:",
            f"                runtime.logger.exception('Failed to evaluate insight %s', {insight_name!r})",
            "                insight_results = {}",
            "    response = ChartResponse(",
            f"        heading={payload.get('heading')!r},",
            f"        title={payload.get('title')!r},",
            f"        source={{'type': {source_type!r}, 'name': {source_name!r}}},",
            f"        chart_type={payload.get('chart_type')!r},",
            f"        x={payload.get('x')!r},",
            f"        y={payload.get('y')!r},",
            f"        color={payload.get('color')!r},",
            "        labels=labels,",
            "        series=series,",
            f"        legend={payload.get('legend') or {}!r},",
            f"        style={payload.get('style') or {}!r},",
            f"        encodings={payload.get('encodings') or {}!r},",
            f"        insight={insight_name!r},",
            "        insights=insight_results,",
            "    )",
            f"    is_reactive = {page.reactive!r} or (dataset.get('reactive') if dataset else False)",
            "    if runtime.REALTIME_ENABLED and is_reactive:",
            f"        await runtime.broadcast_component_update({page.slug!r}, 'chart', {index}, response, meta={meta_expr})",
            "    return response",
        ]
    return []


def _render_form_endpoint(
    page: PageSpec, component: PageComponent, index: int
) -> List[str]:
    """Generate form submission endpoint with validation."""
    payload = component.payload
    slug = page.slug
    base_path = f"/api/pages/{slug}"
    form_id = payload.get("id", f"form_{index}")
    
    # Build validation schema from fields
    fields = payload.get("fields", [])
    validation_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for field in fields:
        field_name = field.get("name")
        component_type = field.get("component", "text_input")
        field_schema = {"type": _infer_json_type(component_type)}
        
        # Add validation constraints
        validation = field.get("validation", {})
        if validation.get("min_length"):
            field_schema["minLength"] = validation["min_length"]
        if validation.get("max_length"):
            field_schema["maxLength"] = validation["max_length"]
        if validation.get("pattern"):
            field_schema["pattern"] = validation["pattern"]
        if validation.get("min_value") is not None:
            field_schema["minimum"] = validation["min_value"]
        if validation.get("max_value") is not None:
            field_schema["maximum"] = validation["max_value"]
        
        validation_schema["properties"][field_name] = field_schema
        
        if field.get("required"):
            validation_schema["required"].append(field_name)
    
    schema_expr = _format_literal(validation_schema)
    submit_action = payload.get("submit_action")
    
    lines = [
        f"@router.post({base_path!r} + '/forms/{form_id}', response_model=Dict[str, Any], tags=['forms'])",
        f"async def {slug}_form_{index}_submit(",
        "    form_data: Dict[str, Any],",
        "    session: AsyncSession = Depends(get_session)",
        ") -> Dict[str, Any]:",
        f"    # Server-side validation using JSON Schema",
        f"    validation_schema = {schema_expr}",
        "    ",
        "    # Validate required fields",
        "    for field in validation_schema.get('required', []):",
        "        if field not in form_data or form_data[field] in (None, '', []):",
        "            raise HTTPException(status_code=400, detail=f'{field} is required')",
        "    ",
        "    # Validate field constraints",
        "    for field_name, field_schema in validation_schema.get('properties', {}).items():",
        "        if field_name not in form_data:",
        "            continue",
        "        value = form_data[field_name]",
        "        ",
        "        # String validation",
        "        if field_schema.get('type') == 'string' and isinstance(value, str):",
        "            if 'minLength' in field_schema and len(value) < field_schema['minLength']:",
        "                raise HTTPException(status_code=400, detail=f'{field_name} must be at least {field_schema[\"minLength\"]} characters')",
        "            if 'maxLength' in field_schema and len(value) > field_schema['maxLength']:",
        "                raise HTTPException(status_code=400, detail=f'{field_name} must be at most {field_schema[\"maxLength\"]} characters')",
        "            if 'pattern' in field_schema:",
        "                import re",
        "                if not re.match(field_schema['pattern'], value):",
        "                    raise HTTPException(status_code=400, detail=f'{field_name} has invalid format')",
        "        ",
        "        # Number validation",
        "        if field_schema.get('type') in ('number', 'integer') and isinstance(value, (int, float)):",
        "            if 'minimum' in field_schema and value < field_schema['minimum']:",
        "                raise HTTPException(status_code=400, detail=f'{field_name} must be at least {field_schema[\"minimum\"]}')",
        "            if 'maximum' in field_schema and value > field_schema['maximum']:",
        "                raise HTTPException(status_code=400, detail=f'{field_name} must be at most {field_schema[\"maximum\"]}')",
        "    ",
        f"    # Execute submit action",
        f"    context = runtime.build_context({page.slug!r})",
        f"    context['form_data'] = form_data",
        "    ",
    ]
    
    if submit_action:
        lines.extend([
            f"    # Call action: {submit_action}",
            f"    action_handler = runtime.ACTION_HANDLERS.get({submit_action!r})",
            "    if action_handler:",
            "        try:",
            "            result = await action_handler(form_data, session, context)",
            "            return {'success': True, 'message': 'Form submitted successfully', 'data': result}",
            "        except Exception as e:",
            "            runtime.logger.exception('Form action %s failed', {submit_action!r})",
            "            raise HTTPException(status_code=500, detail=str(e))",
            "    else:",
            "        runtime.logger.warning('Action handler %s not found', {submit_action!r})",
        ])
    
    lines.extend([
        "    ",
        "    # Default success response",
        "    return {'success': True, 'message': 'Form submitted successfully', 'data': form_data}",
    ])
    
    return lines


def _infer_json_type(component: str) -> str:
    """Infer JSON Schema type from component name."""
    if component in ('checkbox', 'switch'):
        return 'boolean'
    elif component in ('slider', 'number_input'):
        return 'number'
    elif component == 'multiselect':
        return 'array'
    else:
        return 'string'


def _render_insight_endpoint(name: str) -> List[str]:
    return [
        f"@app.get('/api/insights/{name}', response_model=InsightResponse)",
        f"async def insight_{name}() -> InsightResponse:",
        "    context = build_context(None)",
        "    rows: List[Dict[str, Any]] = []",
        f"    result = evaluate_insights_for_dataset({name!r}, rows, context)",
        f"    return InsightResponse(name={name!r}, dataset=result.get('dataset', {name!r}), result=result)",
    ]
