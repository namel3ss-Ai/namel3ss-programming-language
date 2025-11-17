from __future__ import annotations

from textwrap import dedent

CRUD_SECTION = dedent(
    '''

def _crud_sanitize_identifier(value: str) -> str:
    segments = [segment.strip() for segment in (value or "").split(".") if segment and segment.strip()]
    if not segments:
        raise ValueError("Empty identifier path")
    for segment in segments:
        if not _IDENTIFIER_RE.match(segment):
            raise ValueError(f"Invalid identifier segment '{segment}'")
    return ".".join(segments)


def _crud_select_clause(resource: Dict[str, Any]) -> str:
    columns = [column for column in resource.get("select_fields") or [] if column]
    if not columns:
        return "*"
    return ", ".join(_crud_sanitize_identifier(column) for column in columns)


def _crud_allowed_operations(resource: Dict[str, Any]) -> Set[str]:
    allowed = resource.get("allowed_operations") or []
    return {str(operation).lower() for operation in allowed}


def _crud_normalize_pagination(resource: Dict[str, Any], limit: Optional[int], offset: Optional[int]) -> Tuple[int, int]:
    default_limit = int(resource.get("default_limit") or 100)
    max_limit = int(resource.get("max_limit") or max(default_limit, 100))
    limit_value = default_limit if limit is None else max(int(limit), 1)
    if max_limit > 0:
        limit_value = min(limit_value, max_limit)
    offset_value = max(int(offset or 0), 0)
    return limit_value, offset_value


def _crud_build_context(session: Optional[AsyncSession], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = build_context(None)
    if isinstance(base, dict):
        context.update(base)
    if session is not None:
        context["session"] = session
    return context


def _crud_tenant_clause(resource: Dict[str, Any], context: Dict[str, Any], *, required: bool) -> Tuple[str, Dict[str, Any]]:
    tenant_column = resource.get("tenant_column")
    if not tenant_column:
        return "", {}
    tenant_value = context.get("tenant")
    if tenant_value is None and isinstance(context.get("request"), dict):
        tenant_value = context["request"].get("tenant")
    if tenant_value is None:
        if required:
            raise PermissionError("tenant_scope_required")
        return "", {}
    clause = f"{_crud_sanitize_identifier(tenant_column)} = :tenant_value"
    return clause, {"tenant_value": tenant_value}


def _get_crud_resource(slug: str) -> Dict[str, Any]:
    resource = CRUD_RESOURCES.get(slug)
    if not isinstance(resource, dict):
        raise KeyError(slug)
    return resource


def describe_crud_resources() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for slug, spec in CRUD_RESOURCES.items():
        if not isinstance(spec, dict):
            continue
        catalog.append(
            {
                "slug": slug,
                "label": spec.get("label"),
                "source_type": spec.get("source_type"),
                "primary_key": spec.get("primary_key"),
                "tenant_column": spec.get("tenant_column"),
                "operations": list(spec.get("allowed_operations") or []),
                "default_limit": int(spec.get("default_limit") or 0),
                "max_limit": int(spec.get("max_limit") or 0),
                "read_only": bool(spec.get("read_only")),
            }
        )
    catalog.sort(key=lambda item: item["slug"])
    return catalog


async def _crud_table_list(
    resource: Dict[str, Any],
    session: AsyncSession,
    context: Dict[str, Any],
    *,
    limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    select_clause = _crud_select_clause(resource)
    order_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    where_parts: List[str] = []
    params: Dict[str, Any] = dict(tenant_params)
    params.update({"limit": limit, "offset": offset})
    if tenant_clause:
        where_parts.append(tenant_clause)
    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    query = text(f"SELECT {select_clause} FROM {table_name}{where_sql} ORDER BY {order_column} LIMIT :limit OFFSET :offset")
    result = await session.execute(query, params)
    rows = [dict(row) for row in result.mappings()]
    total: Optional[int] = None
    try:
        count_query = text(f"SELECT COUNT(1) FROM {table_name}{where_sql}")
        count_result = await session.execute(count_query, tenant_params)
        total_value = count_result.scalar()
        total = int(total_value) if total_value is not None else None
    except Exception:
        total = None
    return rows, total


async def _crud_dataset_list(
    resource: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    dataset_name = resource.get("source_name")
    rows = await fetch_dataset_rows(dataset_name, session, context)
    tenant_column = resource.get("tenant_column")
    if tenant_column:
        clause, params = _crud_tenant_clause(resource, context, required=True)
        tenant_value = params.get("tenant_value")
        rows = [row for row in rows if str(row.get(tenant_column)) == str(tenant_value)]
    total = len(rows)
    window = rows[offset : offset + limit]
    return window, total


async def _crud_select_table_row(
    resource: Dict[str, Any],
    identifier: Any,
    session: AsyncSession,
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    select_clause = _crud_select_clause(resource)
    pk_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params: Dict[str, Any] = dict(tenant_params)
    params["pk"] = identifier
    where_clause = f"{pk_column} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    query = text(f"SELECT {select_clause} FROM {table_name} WHERE {where_clause} LIMIT 1")
    result = await session.execute(query, params)
    mapping = result.mappings().first()
    return dict(mapping) if mapping else None


async def _crud_select_dataset_row(resource: Dict[str, Any], identifier: Any, session: Optional[AsyncSession], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows, _total = await _crud_dataset_list(resource, session, context, limit=resource.get("max_limit", 1000), offset=0)
    pk_column = resource.get("primary_key") or "id"
    for row in rows:
        if str(row.get(pk_column)) == str(identifier):
            return row
    return None


async def _crud_lookup_row(
    resource: Dict[str, Any],
    identifier: Any,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        return await _crud_select_dataset_row(resource, identifier, session, context)
    if session is None:
        raise RuntimeError("Database session is required for CRUD table operations")
    return await _crud_select_table_row(resource, identifier, session, context)


async def list_crud_resource(
    slug: str,
    session: Optional[AsyncSession],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "list" not in allowed:
        raise PermissionError("list")
    context = _crud_build_context(session)
    limit_value, offset_value = _crud_normalize_pagination(resource, limit, offset)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        rows, total = await _crud_dataset_list(resource, session, context, limit=limit_value, offset=offset_value)
    else:
        if session is None:
            raise RuntimeError("Database session is required for CRUD queries")
        rows, total = await _crud_table_list(resource, session, context, limit=limit_value, offset=offset_value)
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": "ok",
        "items": rows,
        "limit": limit_value,
        "offset": offset_value,
        "total": total,
        "errors": errors,
    }


async def retrieve_crud_resource(
    slug: str,
    identifier: Any,
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "retrieve" not in allowed:
        raise PermissionError("retrieve")
    context = _crud_build_context(session)
    row = await _crud_lookup_row(resource, identifier, session, context)
    errors = _collect_runtime_errors(context)
    status = "ok" if row is not None else "not_found"
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "item": row,
        "errors": errors,
    }


async def create_crud_resource(
    slug: str,
    payload: Dict[str, Any],
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "create" not in allowed or resource.get("read_only"):
        raise PermissionError("create")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    values = dict(payload or {})
    if tenant_params and resource.get("tenant_column") not in values:
        values[resource["tenant_column"]] = tenant_params["tenant_value"]
    mutable_fields = resource.get("mutable_fields") or []
    if not mutable_fields:
        mutable_fields = [field for field in resource.get("select_fields") or [] if field and field != resource.get("primary_key")]
    columns: List[str] = []
    params: Dict[str, Any] = {}
    for field in mutable_fields:
        if field in values:
            columns.append(field)
            params[field] = values[field]
    if not columns:
        raise ValueError("No writable fields provided")
    column_clause = ", ".join(_crud_sanitize_identifier(column) for column in columns)
    placeholder_clause = ", ".join(f":{column}" for column in columns)
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    returning_clause = _crud_select_clause(resource)
    statement = text(f"INSERT INTO {table_name} ({column_clause}) VALUES ({placeholder_clause}) RETURNING {returning_clause}")
    try:
        result = await session.execute(statement, params)
        await session.commit()
        created = result.mappings().first()
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_create_failed",
            message=f"Failed to create record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    if created is None:
        pk_name = resource.get("primary_key") or "id"
        identifier = params.get(pk_name)
        created = await _crud_lookup_row(resource, identifier, session, context) if identifier is not None else params
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": "created",
        "item": dict(created) if created is not None else None,
        "errors": errors,
    }


async def update_crud_resource(
    slug: str,
    identifier: Any,
    payload: Dict[str, Any],
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "update" not in allowed or resource.get("read_only"):
        raise PermissionError("update")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        raise PermissionError("update")
    mutable_fields = resource.get("mutable_fields") or []
    if not mutable_fields:
        mutable_fields = [field for field in resource.get("select_fields") or [] if field and field != resource.get("primary_key")]
    assignments: List[str] = []
    params: Dict[str, Any] = {"pk": identifier}
    for field in mutable_fields:
        if field in payload:
            sanitized = _crud_sanitize_identifier(field)
            placeholder = f"set_{field}"
            assignments.append(f"{sanitized} = :{placeholder}")
            params[placeholder] = payload[field]
    if not assignments:
        raise ValueError("No mutable fields provided")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params.update(tenant_params)
    where_clause = f"{_crud_sanitize_identifier(resource.get('primary_key') or 'id')} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    update_sql = text(f"UPDATE {table_name} SET {', '.join(assignments)} WHERE {where_clause}")
    try:
        result = await session.execute(update_sql, params)
        await session.commit()
        updated_rows = result.rowcount or 0
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_update_failed",
            message=f"Failed to update record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    if updated_rows <= 0:
        status = "not_found"
        item: Optional[Dict[str, Any]] = None
    else:
        status = "updated"
        item = await _crud_lookup_row(resource, identifier, session, context)
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "item": item,
        "errors": errors,
    }


async def delete_crud_resource(
    slug: str,
    identifier: Any,
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "delete" not in allowed or resource.get("read_only"):
        raise PermissionError("delete")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        raise PermissionError("delete")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params: Dict[str, Any] = dict(tenant_params)
    params["pk"] = identifier
    pk_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    where_clause = f"{pk_column} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    delete_sql = text(f"DELETE FROM {table_name} WHERE {where_clause}")
    try:
        result = await session.execute(delete_sql, params)
        await session.commit()
        removed = result.rowcount or 0
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_delete_failed",
            message=f"Failed to delete record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    status = "deleted" if removed > 0 else "not_found"
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "deleted": removed > 0,
        "errors": errors,
    }
'''
).strip()

__all__ = ["CRUD_SECTION"]
