"""
FastAPI routes for {{ entity_name }} CRUD operations.

Provides RESTful HTTP endpoints with OpenAPI documentation.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from models.schemas import (
    {{ entity_name }}Create,
    {{ entity_name }}Update,
    {{ entity_name }}Response,
    {{ entity_name }}List,
    ErrorResponse,
)
from api.dependencies import get_repository, get_settings, get_tenant_id
from repository import {{ entity_name }}Repository
from config.settings import Settings


router = APIRouter(
    prefix="/{{ endpoint_prefix }}",
    tags=["{{ entity_name }}"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


@router.post(
    "/",
    response_model={{ entity_name }}Response,
    status_code=status.HTTP_201_CREATED,
    summary="Create new {{ entity_name | lower }}",
    description="Create a new {{ entity_name | lower }} with the provided data.",
    responses={
        201: {"description": "{{ entity_name }} created successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "{{ entity_name }} already exists"},
    },
)
async def create_item(
    item_data: {{ entity_name }}Create,
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}Response:
    """
    Create a new {{ entity_name | lower }}.
    
    **Request Body:**
    - name: {{ entity_name }} name (required, 1-255 chars)
    - description: Optional description (max 2000 chars)
    - quantity: Available quantity (default: 0, must be >= 0)
    - price: Unit price (default: 0.00, must be >= 0, max 2 decimals)
    - is_active: Active status (default: true)
    - tags: List of tags for categorization (max 20 tags)
    - metadata: Additional key-value metadata
    
    **Returns:**
    - Created {{ entity_name | lower }} with generated ID and timestamps
    """
    from models.domain import {{ entity_name }}
    
    # Create domain entity
    item = {{ entity_name }}(
        name=item_data.name,
        description=item_data.description,
        quantity=item_data.quantity,
        price=item_data.price,
        is_active=item_data.is_active,
        tags=item_data.tags,
        metadata=item_data.metadata,
        tenant_id=tenant_id,
    )
    
    try:
        created_item = await repository.create(item)
        return {{ entity_name }}Response.model_validate(created_item.to_dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get(
    "/{item_id}",
    response_model={{ entity_name }}Response,
    summary="Get {{ entity_name | lower }} by ID",
    description="Retrieve a specific {{ entity_name | lower }} by its unique identifier.",
    responses={
        200: {"description": "{{ entity_name }} found"},
        404: {"model": ErrorResponse, "description": "{{ entity_name }} not found"},
    },
)
async def get_item(
    item_id: UUID,
    include_deleted: bool = Query(False, description="Include soft-deleted items"),
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}Response:
    """
    Get a {{ entity_name | lower }} by ID.
    
    **Path Parameters:**
    - item_id: UUID of the {{ entity_name | lower }}
    
    **Query Parameters:**
    - include_deleted: Whether to include soft-deleted items (default: false)
    
    **Returns:**
    - {{ entity_name }} details if found
    - 404 error if not found
    """
    item = await repository.get_by_id(item_id, include_deleted=include_deleted, tenant_id=tenant_id)
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{{ entity_name }} with ID {item_id} not found",
        )
    
    return {{ entity_name }}Response.model_validate(item.to_dict())


@router.get(
    "/",
    response_model={{ entity_name }}List,
    summary="List {{ entity_name | lower }}s",
    description="List {{ entity_name | lower }}s with pagination and optional filtering.",
    responses={
        200: {"description": "List of {{ entity_name | lower }}s"},
    },
)
async def list_items(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include soft-deleted items"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags (must have ALL)"),
    repository: {{ entity_name }}Repository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}List:
    """
    List {{ entity_name | lower }}s with pagination and filtering.
    
    **Query Parameters:**
    - page: Page number starting from 1 (default: 1)
    - page_size: Items per page, max 100 (default: 20)
    - include_deleted: Include soft-deleted items (default: false)
    - is_active: Filter by active status (default: all)
    - tags: Comma-separated tags, items must have ALL tags
    
    **Returns:**
    - Paginated list of {{ entity_name | lower }}s with metadata
    """
    # Enforce max page size
    page_size = min(page_size, settings.max_page_size)
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    
    items, total = await repository.list_items(
        page=page,
        page_size=page_size,
        include_deleted=include_deleted,
        is_active=is_active,
        tags=tag_list,
        tenant_id=tenant_id,
    )
    
    # Calculate pagination metadata
    has_next = (page * page_size) < total
    has_prev = page > 1
    
    return {{ entity_name }}List(
        items=[{{ entity_name }}Response.model_validate(item.to_dict()) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
        has_prev=has_prev,
    )


@router.get(
    "/search/",
    response_model={{ entity_name }}List,
    summary="Search {{ entity_name | lower }}s by name",
    description="Search {{ entity_name | lower }}s using case-insensitive partial name matching.",
    responses={
        200: {"description": "Search results"},
    },
)
async def search_items(
    q: str = Query(..., min_length=1, max_length=255, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    repository: {{ entity_name }}Repository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}List:
    """
    Search {{ entity_name | lower }}s by name.
    
    **Query Parameters:**
    - q: Search query (required, 1-255 chars)
    - page: Page number (default: 1)
    - page_size: Items per page, max 100 (default: 20)
    
    **Returns:**
    - Paginated search results ordered by name
    """
    page_size = min(page_size, settings.max_page_size)
    
    items, total = await repository.search_by_name(
        query=q,
        page=page,
        page_size=page_size,
        tenant_id=tenant_id,
    )
    
    has_next = (page * page_size) < total
    has_prev = page > 1
    
    return {{ entity_name }}List(
        items=[{{ entity_name }}Response.model_validate(item.to_dict()) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
        has_prev=has_prev,
    )


@router.put(
    "/{item_id}",
    response_model={{ entity_name }}Response,
    summary="Update {{ entity_name | lower }}",
    description="Update an existing {{ entity_name | lower }}. Only provided fields are updated.",
    responses={
        200: {"description": "{{ entity_name }} updated successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "{{ entity_name }} not found"},
    },
)
async def update_item(
    item_id: UUID,
    item_data: {{ entity_name }}Update,
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}Response:
    """
    Update an existing {{ entity_name | lower }}.
    
    **Path Parameters:**
    - item_id: UUID of the {{ entity_name | lower }} to update
    
    **Request Body:**
    - All fields are optional
    - Only provided fields will be updated
    - Omitted fields remain unchanged
    
    **Returns:**
    - Updated {{ entity_name | lower }}
    - 404 error if not found
    """
    # Get existing item
    item = await repository.get_by_id(item_id, tenant_id=tenant_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{{ entity_name }} with ID {item_id} not found",
        )
    
    # Update fields
    update_data = item_data.model_dump(exclude_unset=True)
    item.update_fields(**update_data)
    
    try:
        updated_item = await repository.update(item)
        return {{ entity_name }}Response.model_validate(updated_item.to_dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete {{ entity_name | lower }}",
    description="Delete a {{ entity_name | lower }} (soft delete by default).",
    responses={
        204: {"description": "{{ entity_name }} deleted successfully"},
        404: {"model": ErrorResponse, "description": "{{ entity_name }} not found"},
    },
)
async def delete_item(
    item_id: UUID,
    hard: bool = Query(False, description="Hard delete (permanent)"),
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> None:
    """
    Delete a {{ entity_name | lower }}.
    
    **Path Parameters:**
    - item_id: UUID of the {{ entity_name | lower }} to delete
    
    **Query Parameters:**
    - hard: If true, permanently delete. If false (default), soft delete.
    
    **Returns:**
    - 204 No Content on success
    - 404 error if not found
    """
    deleted = await repository.delete(item_id, soft=not hard, tenant_id=tenant_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{{ entity_name }} with ID {item_id} not found",
        )


@router.post(
    "/{item_id}/restore",
    response_model={{ entity_name }}Response,
    summary="Restore deleted {{ entity_name | lower }}",
    description="Restore a soft-deleted {{ entity_name | lower }}.",
    responses={
        200: {"description": "{{ entity_name }} restored successfully"},
        404: {"model": ErrorResponse, "description": "{{ entity_name }} not found or not deleted"},
    },
)
async def restore_item(
    item_id: UUID,
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> {{ entity_name }}Response:
    """
    Restore a soft-deleted {{ entity_name | lower }}.
    
    **Path Parameters:**
    - item_id: UUID of the {{ entity_name | lower }} to restore
    
    **Returns:**
    - Restored {{ entity_name | lower }}
    - 404 error if not found or not deleted
    """
    restored = await repository.restore(item_id, tenant_id=tenant_id)
    
    if not restored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{{ entity_name }} with ID {item_id} not found or not deleted",
        )
    
    # Fetch and return restored item
    item = await repository.get_by_id(item_id, include_deleted=False, tenant_id=tenant_id)
    return {{ entity_name }}Response.model_validate(item.to_dict())


@router.get(
    "/stats/count",
    summary="Count {{ entity_name | lower }}s",
    description="Get count of {{ entity_name | lower }}s matching criteria.",
    responses={
        200: {"description": "Count result"},
    },
)
async def count_items(
    include_deleted: bool = Query(False, description="Include soft-deleted items"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    repository: {{ entity_name }}Repository = Depends(get_repository),
    tenant_id: Optional[str] = Depends(get_tenant_id),
) -> JSONResponse:
    """
    Count {{ entity_name | lower }}s matching criteria.
    
    **Query Parameters:**
    - include_deleted: Include soft-deleted items (default: false)
    - is_active: Filter by active status (default: all)
    
    **Returns:**
    - JSON with count value
    """
    count = await repository.count(
        include_deleted=include_deleted,
        is_active=is_active,
        tenant_id=tenant_id,
    )
    
    return JSONResponse(content={"count": count})
