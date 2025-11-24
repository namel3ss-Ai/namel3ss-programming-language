"""
Generate FastAPI dataset endpoints for CRUD operations and real-time updates.

This module creates production-ready dataset endpoints that:
- Support pagination, sorting, filtering, and search
- Implement safe CRUD operations with validation
- Integrate with Redis pub/sub for real-time updates
- Include comprehensive error handling and observability
"""

import textwrap
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from namel3ss.ir import DatasetSpec, BackendIR


def python_identifier(name: str) -> str:
    """Convert a name to a valid Python identifier."""
    import re
    # Replace non-alphanumeric characters with underscores
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if identifier and identifier[0].isdigit():
        identifier = f"_{identifier}"
    # Convert to title case for class names
    return ''.join(word.capitalize() for word in identifier.split('_'))


def generate_timestamp() -> str:
    """Generate a timestamp string for documentation."""
    import datetime
    return datetime.datetime.now().isoformat()


def _render_dataset_router_module(backend_ir: "BackendIR") -> str:
    """Generate datasets.py router module with CRUD endpoints."""
    
    # Filter datasets that have data binding enabled
    bindable_datasets = [
        ds for ds in backend_ir.datasets 
        if ds.access_policy and not ds.access_policy.get("read_only", True)
    ]
    
    has_realtime = any(ds.realtime_enabled for ds in backend_ir.datasets)
    
    imports = [
        "from typing import Any, Dict, List, Optional",
        "from fastapi import APIRouter, HTTPException, Query, Depends, status",
        "from pydantic import BaseModel, Field",
        "import asyncio",
        "import logging",
        "",
        "from ..schemas import ErrorResponse",
        "from ..database import get_db_session",
    ]
    
    if has_realtime:
        imports.extend([
            "from ..runtime.realtime import broadcast_dataset_change",
        ])
    
    # Add OpenTelemetry imports for observability
    imports.extend([
        "",
        "# Observability",
        "from opentelemetry import trace",
        "from opentelemetry.trace import Status, StatusCode",
        "",
        "logger = logging.getLogger(__name__)",
        "tracer = trace.get_tracer(__name__)",
    ])
    
    # Generate request/response models for each dataset
    models = []
    endpoint_functions = []
    
    for dataset in backend_ir.datasets:
        dataset_models = _generate_dataset_models(dataset)
        models.append(dataset_models)
        
        # Generate read endpoints (always available)
        read_endpoints = _generate_read_endpoints(dataset)
        endpoint_functions.append(read_endpoints)
        
        # Generate write endpoints if dataset is editable
        if dataset.access_policy and not dataset.access_policy.get("read_only", True):
            write_endpoints = _generate_write_endpoints(dataset, has_realtime)
            endpoint_functions.append(write_endpoints)
    
    router_setup = textwrap.dedent(f'''
        # Router setup
        router = APIRouter(prefix="/api/datasets", tags=["datasets"])
        
        # Common query parameters for pagination and filtering
        class DatasetQuery(BaseModel):
            page: int = Field(default=1, ge=1, description="Page number")
            page_size: int = Field(default=50, ge=1, le=1000, description="Items per page")
            sort_by: Optional[str] = Field(default=None, description="Sort column")
            sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")
            search: Optional[str] = Field(default=None, description="Search term")
            
        class DatasetResponse(BaseModel):
            data: List[Dict[str, Any]]
            total: int
            page: int
            page_size: int
            pages: int
    ''')
    
    module_content = "\\n".join([
        "\\n".join(imports),
        "",
        "\\n".join(models),
        "",
        router_setup,
        "",
        "\\n".join(endpoint_functions),
    ])
    
    return module_content


def _generate_dataset_models(dataset: "DatasetSpec") -> str:
    """Generate Pydantic models for dataset operations."""
    name = python_identifier(dataset.name)
    
    # Generate field definitions from dataset schema
    fields = []
    for field in dataset.schema:
        field_type = _python_type_from_schema(field.dtype)
        nullable = " | None" if field.nullable else ""
        description = f', description="{field.description}"' if field.description else ""
        
        if field.name == dataset.primary_key:
            fields.append(f'    {field.name}: {field_type}{nullable} = Field(default=None{description})')
        else:
            fields.append(f'    {field.name}: {field_type}{nullable}{description}')
    
    return textwrap.dedent(f'''
        # Models for {dataset.name} dataset
        class {name}Item(BaseModel):
        {chr(10).join(fields) if fields else "    pass"}
        
        class {name}CreateRequest(BaseModel):
        {chr(10).join(f"    {f.split(': ')[0]}: {f.split(': ')[1]}" for f in fields if dataset.primary_key not in f)}
        
        class {name}UpdateRequest(BaseModel):
        {chr(10).join(f"    {f.split(': ')[0]}: {f.split(': ')[1]} = None" for f in fields if dataset.primary_key not in f)}
    ''')


def _generate_read_endpoints(dataset: "DatasetSpec") -> str:
    """Generate read-only endpoints for a dataset."""
    name = python_identifier(dataset.name)
    dataset_name = dataset.name
    
    # Determine sortable and filterable fields from access policy
    access_policy = dataset.access_policy or {}
    sortable_fields = access_policy.get("sortable_fields", [])
    filterable_fields = access_policy.get("filterable_fields", [])
    searchable_fields = access_policy.get("searchable_fields", [])
    
    return textwrap.dedent(f'''
        @router.get("/{dataset_name}", response_model=DatasetResponse)
        async def get_{name}_dataset(
            page: int = Query(1, ge=1),
            page_size: int = Query(50, ge=1, le=1000),
            sort_by: Optional[str] = Query(None),
            sort_order: str = Query("asc", pattern="^(asc|desc)$"),
            search: Optional[str] = Query(None),
            db = Depends(get_db_session),
        ):
            """Get paginated {dataset_name} dataset with filtering and sorting."""
            with tracer.start_as_current_span("get_{name}_dataset") as span:
                try:
                    span.set_attribute("dataset.name", "{dataset_name}")
                    span.set_attribute("page", page)
                    span.set_attribute("page_size", page_size)
                    
                    # Execute query
                    from .sql_compiler import compile_dataset_to_sql
                    sql_result = compile_dataset_to_sql(
                        dataset_name="{dataset_name}",
                        page=page,
                        page_size=page_size,
                        sort_by=sort_by if sort_by in {sortable_fields} else None,
                        sort_order=sort_order,
                        search=search if searchable_fields else None,
                        searchable_fields={searchable_fields},
                    )
                    
                    # Execute query
                    result = await db.execute(sql_result["query"], sql_result.get("params", {{}}))
                    rows = result.fetchall()
                    
                    # Get total count for pagination
                    count_result = await db.execute(sql_result["count_query"], sql_result.get("params", {{}}))
                    total = count_result.scalar() or 0
                    
                    # Convert to response format
                    data = [dict(row._mapping) for row in rows]
                    pages = (total + page_size - 1) // page_size
                    
                    span.set_attribute("results.total", total)
                    span.set_attribute("results.returned", len(data))
                    span.set_status(Status(StatusCode.OK))
                    
                    return DatasetResponse(
                        data=data,
                        total=total,
                        page=page,
                        page_size=page_size,
                        pages=pages,
                    )
                    
                except Exception as e:
                    logger.exception(f"Error fetching {dataset_name} dataset")
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to fetch {dataset_name} data"
                    )
    ''')


def _generate_write_endpoints(dataset: "DatasetSpec", has_realtime: bool) -> str:
    """Generate CRUD endpoints for an editable dataset."""
    name = python_identifier(dataset.name)
    dataset_name = dataset.name
    primary_key = dataset.primary_key or "id"
    
    access_policy = dataset.access_policy or {}
    allow_create = access_policy.get("allow_create", False)
    allow_update = access_policy.get("allow_update", False)
    allow_delete = access_policy.get("allow_delete", False)
    
    endpoints = []
    
    if allow_create:
        create_endpoint = textwrap.dedent(f'''
            @router.post("/{dataset_name}", response_model={name}Item, status_code=status.HTTP_201_CREATED)
            async def create_{name}_item(
                item: {name}CreateRequest,
                db = Depends(get_db_session),
            ):
                """Create a new {dataset_name} record."""
                with tracer.start_as_current_span("create_{name}_item") as span:
                    try:
                        span.set_attribute("dataset.name", "{dataset_name}")
                        
                        # Insert record using SQL compiler
                        from .sql_compiler import compile_dataset_insert
                        insert_result = compile_dataset_insert(
                            dataset_name="{dataset_name}",
                            data=item.dict(exclude_unset=True),
                        )
                        
                        result = await db.execute(insert_result["query"], insert_result["params"])
                        await db.commit()
                        
                        # Get the created record
                        created_id = result.inserted_primary_key[0]
                        span.set_attribute("record.{primary_key}", str(created_id))
                        
                        # Fetch the complete created record
                        select_result = await db.execute(
                            insert_result["select_query"], 
                            {{"{primary_key}": created_id}}
                        )
                        created_record = select_result.fetchone()
                        
                        if not created_record:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to retrieve created record"
                            )
                        
                        record_data = dict(created_record._mapping)
                        
                        # Broadcast change for realtime updates
                        if has_realtime:
                            try:
                                await broadcast_dataset_change(
                                    dataset_name="{dataset_name}",
                                    event_type="create",
                                    data=record_data,
                                )
                            except Exception as broadcast_error:
                                logger.warning("Failed to broadcast create event: %s", broadcast_error)
                        
                        span.set_status(Status(StatusCode.OK))
                        return {name}Item(**record_data)
                        
                    except Exception as e:
                        await db.rollback()
                        logger.exception(f"Error creating {dataset_name} record")
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to create {dataset_name} record"
                        )
        ''')
        endpoints.append(create_endpoint)
    
    if allow_update:
        update_endpoint = textwrap.dedent(f'''
            @router.patch("/{dataset_name}/{{{primary_key}}}", response_model={name}Item)
            async def update_{name}_item(
                {primary_key}: str,
                item: {name}UpdateRequest,
                db = Depends(get_db_session),
            ):
                """Update an existing {dataset_name} record."""
                with tracer.start_as_current_span("update_{name}_item") as span:
                    try:
                        span.set_attribute("dataset.name", "{dataset_name}")
                        span.set_attribute("record.{primary_key}", {primary_key})
                        
                        # Update record using SQL compiler
                        from .sql_compiler import compile_dataset_update
                        update_result = compile_dataset_update(
                            dataset_name="{dataset_name}",
                            record_id={primary_key},
                            data=item.dict(exclude_unset=True, exclude_none=True),
                        )
                        
                        result = await db.execute(update_result["query"], update_result["params"])
                        
                        if result.rowcount == 0:
                            raise HTTPException(
                                status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"{dataset_name} record not found"
                            )
                        
                        await db.commit()
                        
                        # Fetch the updated record
                        select_result = await db.execute(
                            update_result["select_query"], 
                            {{"{primary_key}": {primary_key}}}
                        )
                        updated_record = select_result.fetchone()
                        
                        record_data = dict(updated_record._mapping)
                        
                        # Broadcast change for realtime updates  
                        if has_realtime:
                            try:
                                await broadcast_dataset_change(
                                    dataset_name="{dataset_name}",
                                    event_type="update",
                                    data=record_data,
                                )
                            except Exception as broadcast_error:
                                logger.warning("Failed to broadcast update event: %s", broadcast_error)
                        
                        span.set_status(Status(StatusCode.OK))
                        return {name}Item(**record_data)
                        
                    except HTTPException:
                        await db.rollback()
                        raise
                    except Exception as e:
                        await db.rollback()
                        logger.exception(f"Error updating {dataset_name} record")
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to update {dataset_name} record"
                        )
        ''')
        endpoints.append(update_endpoint)
    
    if allow_delete:
        delete_endpoint = textwrap.dedent(f'''
            @router.delete("/{dataset_name}/{{{primary_key}}}", status_code=status.HTTP_204_NO_CONTENT)
            async def delete_{name}_item(
                {primary_key}: str,
                db = Depends(get_db_session),
            ):
                """Delete a {dataset_name} record."""
                with tracer.start_as_current_span("delete_{name}_item") as span:
                    try:
                        span.set_attribute("dataset.name", "{dataset_name}")
                        span.set_attribute("record.{primary_key}", {primary_key})
                        
                        # Delete record using SQL compiler
                        from .sql_compiler import compile_dataset_delete
                        delete_result = compile_dataset_delete(
                            dataset_name="{dataset_name}",
                            record_id={primary_key},
                        )
                        
                        result = await db.execute(delete_result["query"], delete_result["params"])
                        
                        if result.rowcount == 0:
                            raise HTTPException(
                                status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"{dataset_name} record not found"
                            )
                        
                        await db.commit()
                        
                        # Broadcast change for realtime updates
                        if has_realtime:
                            try:
                                await broadcast_dataset_change(
                                    dataset_name="{dataset_name}",
                                    event_type="delete", 
                                    data={{"{primary_key}": {primary_key}}},
                                )
                            except Exception as broadcast_error:
                                logger.warning("Failed to broadcast delete event: %s", broadcast_error)
                        
                        span.set_status(Status(StatusCode.OK))
                        
                    except HTTPException:
                        await db.rollback()
                        raise
                    except Exception as e:
                        await db.rollback()
                        logger.exception(f"Error deleting {dataset_name} record")
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to delete {dataset_name} record"
                        )
        ''')
        endpoints.append(delete_endpoint)
    
    return "\\n\\n".join(endpoints)


def _generate_realtime_broadcast(event_type: str, data_var: str, dataset_name: str) -> str:
    """Generate code to broadcast dataset changes via realtime channels."""
    return textwrap.dedent(f'''
        # Broadcast change for realtime updates
        try:
            await broadcast_dataset_change(
                dataset_name="{dataset_name}",
                event_type="{event_type}",
                data={data_var},
            )
        except Exception as broadcast_error:
            logger.warning("Failed to broadcast {event_type} event: %s", broadcast_error)
    ''').strip()


def _python_type_from_schema(dtype: str) -> str:
    """Convert dataset schema type to Python type annotation."""
    type_mapping = {
        "string": "str",
        "integer": "int", 
        "float": "float",
        "boolean": "bool",
        "datetime": "str",  # ISO format string
        "date": "str",      # ISO format string
        "json": "Dict[str, Any]",
        "array": "List[Any]",
    }
    return type_mapping.get(dtype.lower(), "str")