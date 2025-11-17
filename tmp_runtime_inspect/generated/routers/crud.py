"""Generated FastAPI router for CRUD resources."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies
from ..schemas import (
    CrudCatalogResponse,
    CrudDeleteResponse,
    CrudItemResponse,
    CrudListResponse,
)

router = APIRouter(prefix="/api/crud", tags=["crud"], dependencies=router_dependencies())


def _crud_not_found(slug: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"CRUD resource '{slug}' is not registered.")


def _crud_operation_forbidden(slug: str, operation: str) -> HTTPException:
    return HTTPException(status_code=403, detail=f"Operation '{operation}' is not allowed for resource '{slug}'.")


def _route(method: str, *args, **kwargs):
    """Return a router decorator, falling back when FastAPI stand-ins lack HTTP verbs."""
    decorator = getattr(router, method, None)
    if callable(decorator):
        return decorator(*args, **kwargs)
    fallback = router.post if method != "get" and hasattr(router, "post") else router.get
    return fallback(*args, **kwargs)


@router.get("/", response_model=CrudCatalogResponse)
async def list_crud_resources() -> CrudCatalogResponse:
    resources = runtime.describe_crud_resources()
    return CrudCatalogResponse(resources=resources)


@router.get("/{slug}", response_model=CrudListResponse)
async def list_crud_items(
    slug: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
) -> CrudListResponse:
    try:
        payload = await runtime.list_crud_resource(slug, session, limit=limit, offset=offset)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "list")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudListResponse(**payload)


@router.get("/{slug}/{identifier}", response_model=CrudItemResponse)
async def get_crud_item(
    slug: str,
    identifier: str,
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        payload = await runtime.retrieve_crud_resource(slug, identifier, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "retrieve")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    result = CrudItemResponse(**payload)
    if result.status == "not_found":
        raise HTTPException(404, detail=f"Record '{identifier}' was not found for resource '{slug}'.")
    return result


@router.post("/{slug}", response_model=CrudItemResponse, status_code=201)
async def create_crud_item(
    slug: str,
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        result_payload = await runtime.create_crud_resource(slug, payload, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "create")
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudItemResponse(**result_payload)


@_route("put", "/{slug}/{identifier}", response_model=CrudItemResponse)
@_route("patch", "/{slug}/{identifier}", response_model=CrudItemResponse)
async def update_crud_item(
    slug: str,
    identifier: str,
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        result_payload = await runtime.update_crud_resource(slug, identifier, payload, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "update")
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    result = CrudItemResponse(**result_payload)
    if result.status == "not_found":
        raise HTTPException(404, detail=f"Record '{identifier}' was not found for resource '{slug}'.")
    return result


@_route("delete", "/{slug}/{identifier}", response_model=CrudDeleteResponse)
async def delete_crud_item(
    slug: str,
    identifier: str,
    session: AsyncSession = Depends(get_session),
) -> CrudDeleteResponse:
    try:
        result_payload = await runtime.delete_crud_resource(slug, identifier, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "delete")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    result = CrudDeleteResponse(**result_payload)
    if result.status == "not_found":
        raise HTTPException(404, detail=f"Record '{identifier}' was not found for resource '{slug}'.")
    return result


__all__ = ["router"]
