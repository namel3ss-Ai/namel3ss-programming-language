"""FastAPI application entry point for the Namel3ss generated backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
try:  # FastAPI exposes PlainTextResponse in newer versions
    from fastapi.responses import PlainTextResponse
except ImportError:  # pragma: no cover - fallback for minimal stubs
    try:
        from starlette.responses import PlainTextResponse  # type: ignore
    except ImportError:  # pragma: no cover - final fallback
        class PlainTextResponse(JSONResponse):  # type: ignore
            def __init__(self, content: str, media_type: str = "text/plain", status_code: int = 200) -> None:
                super().__init__(content=content, media_type=media_type, status_code=status_code)

from .generated import runtime
from .generated.helpers import include_generated_routers

_PRODUCTION_MODE = getattr(runtime, "is_production_mode", lambda: False)()

_configure_logging = getattr(runtime, "configure_logging", None)
if callable(_configure_logging):
    _configure_logging()

app = FastAPI(
    title=runtime.APP.get("name", "Namel3ss App"),
    version=str(runtime.APP.get("version", "0.1")),
    docs_url=None if _PRODUCTION_MODE else "/docs",
    redoc_url=None if _PRODUCTION_MODE else "/redoc",
    openapi_url=None if _PRODUCTION_MODE else "/openapi.json",
)

_configure_tracing = getattr(runtime, "configure_tracing", None)
if callable(_configure_tracing):
    _configure_tracing(app)


def _load_user_routes() -> None:
    """Load optional custom routes without failing generation."""

    module_candidates = [
        "custom.routes.custom_api",
        "custom.routes.extensions",
        "custom_api",
        "extensions",
    ]
    for module_name in module_candidates:
        try:
            module = __import__(f"{__package__}.{module_name}", fromlist=["*"])
        except Exception:
            try:
                module = __import__(module_name, fromlist=["*"])
            except Exception:
                continue
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)
        setup = getattr(module, "setup", None)
        if callable(setup):
            try:
                setup(app)
            except Exception:  # pragma: no cover - user extension failure
                runtime.logger.exception("User API setup failed for %s", module_name)


include_generated_routers(app)
_load_user_routes()


@app.middleware("http")
async def _namel3ss_security_layer(request: Request, call_next):
    headers = getattr(request, "headers", {}) or {}
    cookies = getattr(request, "cookies", {}) or {}
    request_id = runtime.ensure_request_id(headers)
    if hasattr(request, "state"):
        setattr(request.state, "namel3ss_request_id", request_id)
    _bind_request_id = getattr(runtime, "bind_request_id", None)
    if callable(_bind_request_id):
        _bind_request_id(request_id)
    _merge_request_context = getattr(runtime, "merge_request_context", None)
    if callable(_merge_request_context):
        try:
            _merge_request_context({"request_id": request_id})
        except Exception:
            runtime.logger.debug("Unable to seed request context", exc_info=True)
    csrf_cookie, should_set_cookie = runtime.ensure_csrf_cookie(cookies)
    started = runtime.request_timer()
    response: Optional[Any] = None
    try:
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - exercised via runtime tests
            status_code, payload = runtime.format_error_response(exc, request_id=request_id)
            response = JSONResponse(status_code=status_code, content=payload)
        status_value = getattr(response, "status_code", 500)
        runtime.apply_security_headers(response, request_id)
        if should_set_cookie and csrf_cookie:
            runtime.set_csrf_cookie(response, csrf_cookie)
        response.headers.setdefault("x-request-id", request_id)
        try:
            client_host = request.client.host if request.client else None
            runtime.record_request_observation(
                started,
                str(request.url.path),
                request.method,
                status_value,
                request_id=request_id,
                client_host=client_host,
            )
        except Exception:  # pragma: no cover - defensive
            runtime.logger.debug("Failed to record request observability", exc_info=True)
        return response
    finally:
        _clear_request_id = getattr(runtime, "clear_request_id", None)
        if callable(_clear_request_id):
            try:
                _clear_request_id()
            except Exception:
                runtime.logger.debug("Unable to clear request identifier", exc_info=True)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return runtime.health_summary()


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return runtime.health_summary()


@app.get("/readyz")
async def readyz() -> Dict[str, Any]:
    return await runtime.readiness_checks()


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    payload = runtime.render_prometheus_metrics()
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")


# Re-export runtime helpers for convenience in tests and extensions.
predict = runtime.predict
predict_model = runtime.predict_model
call_python_model = runtime.call_python_model


class _ConnectorResult(dict):
    """Dictionary wrapper that remains comparable to plain text payloads."""

    def __init__(self, payload: Any) -> None:
        if isinstance(payload, dict):
            super().__init__(payload)
        else:
            super().__init__()
            if payload is not None:
                self["text"] = str(payload)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return self.get("text") == other
        return dict.__eq__(self, other)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"_ConnectorResult({dict.__repr__(self)})"


def call_llm_connector(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Invoke the runtime connector and flatten common response fields for tests."""

    response = runtime.call_llm_connector(name, payload or {})
    if not isinstance(response, dict):
        return response

    flattened: Dict[str, Any] = dict(response)
    result_section = flattened.get("result")
    if result_section is not None:
        wrapper = _ConnectorResult(result_section)
        json_payload = wrapper.get("json") if isinstance(wrapper.get("json"), dict) else None
        if isinstance(json_payload, dict):
            usage = json_payload.get("usage")
            if usage is not None:
                flattened["usage"] = usage
        flattened["result"] = wrapper
    elif "result" in flattened:
        flattened.pop("result", None)
    return flattened


run_chain = runtime.run_chain
run_prompt = runtime.run_prompt
evaluate_experiment = runtime.evaluate_experiment
run_experiment = runtime.run_experiment
run_training_job = runtime.run_training_job
run_tuning_job = runtime.run_tuning_job
resolve_training_job_plan = runtime.resolve_training_job_plan
list_training_jobs = runtime.list_training_jobs
list_tuning_jobs = runtime.list_tuning_jobs
get_training_job = runtime.get_training_job
get_tuning_job = runtime.get_tuning_job
training_job_history = runtime.training_job_history
tuning_job_history = runtime.tuning_job_history
available_training_backends = runtime.available_training_backends
DATASETS = runtime.DATASETS
INSIGHTS = runtime.INSIGHTS
CONTEXT = runtime.CONTEXT
build_context = runtime.build_context
_resolve_connector = runtime._resolve_connector
_run_insight = runtime._run_insight

__all__ = [
    "app",
    "predict",
    "predict_model",
    "call_python_model",
    "call_llm_connector",
    "run_chain",
    "run_prompt",
    "evaluate_experiment",
    "run_experiment",
    "run_training_job",
    "run_tuning_job",
    "resolve_training_job_plan",
    "list_training_jobs",
    "list_tuning_jobs",
    "get_training_job",
    "get_tuning_job",
    "training_job_history",
    "tuning_job_history",
    "available_training_backends",
    "DATASETS",
    "INSIGHTS",
    "CONTEXT",
    "build_context",
    "_resolve_connector",
    "_run_insight",
]
