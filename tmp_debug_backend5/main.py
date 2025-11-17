"""FastAPI application entry point for the Namel3ss generated backend."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI

from .generated import runtime
from .generated.helpers import include_generated_routers

app = FastAPI(
    title=runtime.APP.get("name", "Namel3ss App"),
    version=str(runtime.APP.get("version", "0.1")),
)


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


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# Re-export runtime helpers for convenience in tests and extensions.
predict = runtime.predict
predict_model = runtime.predict_model
call_python_model = runtime.call_python_model
call_llm_connector = runtime.call_llm_connector
run_chain = runtime.run_chain
evaluate_experiment = runtime.evaluate_experiment
run_experiment = runtime.run_experiment
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
    "evaluate_experiment",
    "run_experiment",
    "DATASETS",
    "INSIGHTS",
    "CONTEXT",
    "build_context",
    "_resolve_connector",
    "_run_insight",
]
