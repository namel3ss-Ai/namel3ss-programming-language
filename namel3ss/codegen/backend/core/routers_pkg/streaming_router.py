"""Code generation for streaming LLM endpoints."""

from textwrap import dedent


def _render_streaming_router_module() -> str:
    """Generate FastAPI router for streaming LLM responses."""
    template = '''
"""Generated FastAPI router for streaming LLM endpoints."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import stream_llm_connector, stream_chain

router = APIRouter(tags=["streaming"], dependencies=router_dependencies())


@router.post(
    "/api/llm/{connector}/stream",
    dependencies=[rate_limit_dependency("ai")],
)
async def stream_llm_endpoint(connector: str, payload: Dict[str, Any]):
    """Stream LLM connector responses as Server-Sent Events."""
    
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in stream_llm_connector(connector, payload):
                # Format as SSE
                data = json.dumps(chunk)
                yield f"data: {data}\\n\\n"
        except Exception as exc:
            error_data = json.dumps({"error": str(exc), "type": type(exc).__name__})
            yield f"data: {error_data}\\n\\n"
        finally:
            yield "data: [DONE]\\n\\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post(
    "/api/chains/{chain_name}/stream",
    dependencies=[rate_limit_dependency("ai")],
)
async def stream_chain_endpoint(chain_name: str, payload: Dict[str, Any]):
    """Stream chain execution with step-by-step updates."""
    
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in stream_chain(chain_name, payload):
                data = json.dumps(event)
                yield f"data: {data}\\n\\n"
        except Exception as exc:
            error_data = json.dumps({"error": str(exc), "type": type(exc).__name__})
            yield f"data: {error_data}\\n\\n"
        finally:
            yield "data: [DONE]\\n\\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
'''
    return dedent(template).strip() + "\n"


__all__ = ['_render_streaming_router_module']
