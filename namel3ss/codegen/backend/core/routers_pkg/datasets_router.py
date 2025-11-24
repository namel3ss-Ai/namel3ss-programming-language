"""Generate FastAPI router for dynamic dataset CRUD operations."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.ir import BackendIR, DatasetSpec, EndpointSpec


def _render_datasets_router_module(backend_ir: "BackendIR") -> str:
    """Generate datasets router with CRUD endpoints from BackendIR.
    
    This generates production-grade FastAPI endpoints for dataset CRUD operations:
    - GET /api/datasets/{name} - List with pagination, sorting, filtering
    - POST /api/datasets/{name} - Create new record
    - PATCH /api/datasets/{name}/{id} - Update existing record
    - DELETE /api/datasets/{name}/{id} - Delete record
    
    Security, validation, and realtime event emission included.
    """
    
    # Generate endpoint functions for each dataset
    endpoint_functions = []
    
    for endpoint_spec in backend_ir.endpoints:
        if not endpoint_spec.path.startswith("/api/datasets/"):
            continue
        
        func_code = _generate_endpoint_function(endpoint_spec, backend_ir)
        endpoint_functions.append(func_code)
    
    # Generate router imports and setup
    template = '''
"""Generated FastAPI router for dataset CRUD operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies

router = APIRouter(prefix="/api/datasets", tags=["datasets"], dependencies=router_dependencies())


# ============================================================================
# Pagination & filtering helpers
# ============================================================================

class PaginatedResponse(BaseModel):
    """Standard paginated response format."""
    data: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_more: bool


def _apply_pagination(query, page: int, page_size: int):
    """Apply pagination to SQLAlchemy query."""
    offset = (page - 1) * page_size
    return query.limit(page_size).offset(offset)


def _apply_sorting(query, model, sort_by: Optional[str], sort_order: str):
    """Apply sorting to SQLAlchemy query."""
    if not sort_by:
        return query
    
    field = getattr(model, sort_by, None)
    if field is None:
        return query
    
    if sort_order.lower() == "desc":
        return query.order_by(field.desc())
    return query.order_by(field.asc())


def _apply_filters(query, model, filters: Dict[str, Any]):
    """Apply filters to SQLAlchemy query."""
    if not filters:
        return query
    
    conditions = []
    for field_name, value in filters.items():
        field = getattr(model, field_name, None)
        if field is None:
            continue
        
        # Support different filter operators
        if isinstance(value, dict):
            if "$eq" in value:
                conditions.append(field == value["$eq"])
            elif "$ne" in value:
                conditions.append(field != value["$ne"])
            elif "$gt" in value:
                conditions.append(field > value["$gt"])
            elif "$gte" in value:
                conditions.append(field >= value["$gte"])
            elif "$lt" in value:
                conditions.append(field < value["$lt"])
            elif "$lte" in value:
                conditions.append(field <= value["$lte"])
            elif "$in" in value:
                conditions.append(field.in_(value["$in"]))
            elif "$like" in value:
                conditions.append(field.like(f"%{value['$like']}%"))
        else:
            # Simple equality filter
            conditions.append(field == value)
    
    if conditions:
        return query.where(and_(*conditions))
    return query


# ============================================================================
# Realtime event emission
# ============================================================================

async def _emit_dataset_change(dataset_name: str, event_type: str, data: Dict[str, Any]):
    """Emit dataset change event for realtime subscriptions.
    
    This function checks if realtime is enabled and emits events to Redis pub/sub.
    Gracefully degrades if Redis is not available.
    """
    try:
        realtime_enabled = getattr(runtime, "realtime_enabled", False)
        if not realtime_enabled:
            return
        
        emit_func = getattr(runtime, "emit_dataset_change", None)
        if emit_func and callable(emit_func):
            await emit_func(dataset_name, event_type, data)
    except Exception as e:
        # Log but don't fail the request if event emission fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to emit dataset change event: {e}")


# ============================================================================
# Dataset CRUD endpoints
# ============================================================================

{endpoint_functions}


__all__ = ["router"]
'''
    
    functions_code = "\n\n".join(endpoint_functions)
    return textwrap.dedent(template).strip().format(endpoint_functions=functions_code) + "\n"


def _generate_endpoint_function(endpoint_spec: "EndpointSpec", backend_ir: "BackendIR") -> str:
    """Generate a single endpoint function from EndpointSpec."""
    
    # Parse dataset name from path
    path_parts = endpoint_spec.path.split("/")
    if len(path_parts) < 4:
        return ""
    
    dataset_name = path_parts[3]
    
    # Find the dataset spec
    dataset_spec = None
    for ds in backend_ir.datasets:
        if ds.name == dataset_name:
            dataset_spec = ds
            break
    
    if not dataset_spec:
        return ""
    
    method = endpoint_spec.method.upper()
    
    if method == "GET":
        return _generate_list_endpoint(dataset_name, dataset_spec, endpoint_spec)
    elif method == "POST":
        return _generate_create_endpoint(dataset_name, dataset_spec, endpoint_spec)
    elif method == "PATCH":
        return _generate_update_endpoint(dataset_name, dataset_spec, endpoint_spec)
    elif method == "DELETE":
        return _generate_delete_endpoint(dataset_name, dataset_spec, endpoint_spec)
    
    return ""


def _generate_list_endpoint(dataset_name: str, dataset_spec: "DatasetSpec", endpoint_spec: "EndpointSpec") -> str:
    """Generate GET endpoint for listing dataset records with pagination."""
    
    func_name = f"list_{dataset_name}_records"
    auth_decorator = '@router.get(f"/{dataset_name}", dependencies=[Depends(require_auth)])' if endpoint_spec.auth_required else f'@router.get("/{dataset_name}")'
    
    template = '''
{auth_decorator}
async def {func_name}(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
    search: Optional[str] = Query(None, description="Search query"),
    session: AsyncSession = Depends(get_session),
) -> PaginatedResponse:
    """List {dataset_name} records with pagination, sorting, and filtering.
    
    Security: {security_note}
    """
    
    # Get SQLAlchemy model from runtime
    model = runtime.get_dataset_model("{dataset_name}")
    if model is None:
        raise HTTPException(404, detail="Dataset not found")
    
    # Build base query
    query = select(model)
    
    # Apply search if provided
    if search:
        # Simple text search across all string fields
        search_conditions = []
        for col in model.__table__.columns:
            if str(col.type) in ["VARCHAR", "TEXT", "String"]:
                search_conditions.append(col.like(f"%{{search}}%"))
        if search_conditions:
            query = query.where(or_(*search_conditions))
    
    # Apply sorting
    query = _apply_sorting(query, model, sort_by, sort_order)
    
    # Get total count
    count_query = select(func.count()).select_from(model)
    if search:
        # Reapply search conditions to count query
        search_conditions = []
        for col in model.__table__.columns:
            if str(col.type) in ["VARCHAR", "TEXT", "String"]:
                search_conditions.append(col.like(f"%{{search}}%"))
        if search_conditions:
            count_query = count_query.where(or_(*search_conditions))
    
    result = await session.execute(count_query)
    total = result.scalar() or 0
    
    # Apply pagination
    query = _apply_pagination(query, page, page_size)
    
    # Execute query
    result = await session.execute(query)
    records = result.scalars().all()
    
    # Convert to dictionaries
    data = [{{col.name: getattr(record, col.name) for col in model.__table__.columns}} for record in records]
    
    return PaginatedResponse(
        data=data,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
'''
    
    security_note = "Requires authentication" if endpoint_spec.auth_required else "Public endpoint"
    if endpoint_spec.allowed_capabilities:
        security_note += f", capabilities: {', '.join(endpoint_spec.allowed_capabilities)}"
    
    return textwrap.dedent(template).strip().format(
        auth_decorator=auth_decorator,
        func_name=func_name,
        dataset_name=dataset_name,
        security_note=security_note,
    )


def _generate_create_endpoint(dataset_name: str, dataset_spec: "DatasetSpec", endpoint_spec: "EndpointSpec") -> str:
    """Generate POST endpoint for creating dataset records."""
    
    func_name = f"create_{dataset_name}_record"
    auth_decorator = '@router.post(f"/{dataset_name}", dependencies=[Depends(require_auth)])' if endpoint_spec.auth_required else f'@router.post("/{dataset_name}")'
    
    template = '''
{auth_decorator}
async def {func_name}(
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Create a new {dataset_name} record.
    
    Security: {security_note}
    """
    
    # Get SQLAlchemy model
    model = runtime.get_dataset_model("{dataset_name}")
    if model is None:
        raise HTTPException(404, detail="Dataset not found")
    
    # Validate required fields
    required_fields = {required_fields_list}
    missing_fields = [f for f in required_fields if f not in payload]
    if missing_fields:
        raise HTTPException(400, detail=f"Missing required fields: {{', '.join(missing_fields)}}")
    
    # Create record
    try:
        record = model(**payload)
        session.add(record)
        await session.commit()
        await session.refresh(record)
        
        # Convert to dict
        result = {{col.name: getattr(record, col.name) for col in model.__table__.columns}}
        
        # Emit realtime event
        await _emit_dataset_change("{dataset_name}", "create", result)
        
        return result
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to create record: {{str(e)}}")
'''
    
    # Extract required fields from schema
    required_fields = [f["name"] for f in dataset_spec.schema if f.get("required", False)]
    required_fields_str = str(required_fields)
    
    security_note = "Requires authentication" if endpoint_spec.auth_required else "Public endpoint"
    if endpoint_spec.allowed_capabilities:
        security_note += f", capabilities: {', '.join(endpoint_spec.allowed_capabilities)}"
    
    return textwrap.dedent(template).strip().format(
        auth_decorator=auth_decorator,
        func_name=func_name,
        dataset_name=dataset_name,
        security_note=security_note,
        required_fields_list=required_fields_str,
    )


def _generate_update_endpoint(dataset_name: str, dataset_spec: "DatasetSpec", endpoint_spec: "EndpointSpec") -> str:
    """Generate PATCH endpoint for updating dataset records."""
    
    func_name = f"update_{dataset_name}_record"
    primary_key = dataset_spec.primary_key or "id"
    auth_decorator = f'@router.patch("/{dataset_name}/{{{primary_key}}}", dependencies=[Depends(require_auth)])' if endpoint_spec.auth_required else f'@router.patch("/{dataset_name}/{{{primary_key}}}")'
    
    template = '''
{auth_decorator}
async def {func_name}(
    {primary_key}: str,
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Update an existing {dataset_name} record.
    
    Security: {security_note}
    """
    
    # Get SQLAlchemy model
    model = runtime.get_dataset_model("{dataset_name}")
    if model is None:
        raise HTTPException(404, detail="Dataset not found")
    
    # Find record
    query = select(model).where(getattr(model, "{primary_key}") == {primary_key})
    result = await session.execute(query)
    record = result.scalar_one_or_none()
    
    if record is None:
        raise HTTPException(404, detail="Record not found")
    
    # Update fields
    try:
        for key, value in payload.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        await session.commit()
        await session.refresh(record)
        
        # Convert to dict
        result = {{col.name: getattr(record, col.name) for col in model.__table__.columns}}
        
        # Emit realtime event
        await _emit_dataset_change("{dataset_name}", "update", result)
        
        return result
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to update record: {{str(e)}}")
'''
    
    security_note = "Requires authentication" if endpoint_spec.auth_required else "Public endpoint"
    if endpoint_spec.allowed_capabilities:
        security_note += f", capabilities: {', '.join(endpoint_spec.allowed_capabilities)}"
    
    return textwrap.dedent(template).strip().format(
        auth_decorator=auth_decorator,
        func_name=func_name,
        dataset_name=dataset_name,
        primary_key=primary_key,
        security_note=security_note,
    )


def _generate_delete_endpoint(dataset_name: str, dataset_spec: "DatasetSpec", endpoint_spec: "EndpointSpec") -> str:
    """Generate DELETE endpoint for deleting dataset records."""
    
    func_name = f"delete_{dataset_name}_record"
    primary_key = dataset_spec.primary_key or "id"
    auth_decorator = f'@router.delete("/{dataset_name}/{{{primary_key}}}", dependencies=[Depends(require_auth)])' if endpoint_spec.auth_required else f'@router.delete("/{dataset_name}/{{{primary_key}}}")'
    
    template = '''
{auth_decorator}
async def {func_name}(
    {primary_key}: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Delete a {dataset_name} record.
    
    Security: {security_note}
    """
    
    # Get SQLAlchemy model
    model = runtime.get_dataset_model("{dataset_name}")
    if model is None:
        raise HTTPException(404, detail="Dataset not found")
    
    # Find record
    query = select(model).where(getattr(model, "{primary_key}") == {primary_key})
    result = await session.execute(query)
    record = result.scalar_one_or_none()
    
    if record is None:
        raise HTTPException(404, detail="Record not found")
    
    # Convert to dict before deletion (for event emission)
    record_data = {{col.name: getattr(record, col.name) for col in model.__table__.columns}}
    
    # Delete record
    try:
        await session.delete(record)
        await session.commit()
        
        # Emit realtime event
        await _emit_dataset_change("{dataset_name}", "delete", record_data)
        
        return {{"success": True, "deleted_id": {primary_key}}}
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to delete record: {{str(e)}}")
'''
    
    security_note = "Requires authentication" if endpoint_spec.auth_required else "Public endpoint"
    if endpoint_spec.allowed_capabilities:
        security_note += f", capabilities: {', '.join(endpoint_spec.allowed_capabilities)}"
    
    return textwrap.dedent(template).strip().format(
        auth_decorator=auth_decorator,
        func_name=func_name,
        dataset_name=dataset_name,
        primary_key=primary_key,
        security_note=security_note,
    )
