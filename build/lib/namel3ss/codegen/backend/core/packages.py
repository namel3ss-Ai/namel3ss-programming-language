"""Render helper packages and custom extension stubs."""

from __future__ import annotations

import textwrap

__all__ = [
    "_render_generated_package",
    "_render_helpers_package",
    "_render_custom_readme",
    "_render_custom_api_stub",
]


def _render_generated_package() -> str:
    template = '''
"""Generated backend package for Namel3ss (N3).

This file is created automatically. Manual edits may be overwritten.
"""

from __future__ import annotations

from . import runtime

__all__ = ["runtime"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_helpers_package() -> str:
    template = '''
"""Helper utilities for Namel3ss generated routers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException

try:  # FastAPI >=0.99 exposes Request at top-level
    from fastapi import Request  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - allow older FastAPI versions
    try:
        from starlette.requests import Request  # type: ignore
    except ImportError:  # pragma: no cover - minimal fallback for tests
        class Request:  # type: ignore
            headers: Dict[str, str]
            state: Any

            def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
                self.headers = headers or {}
                self.state = type("State", (), {})()

try:  # pragma: no cover - runtime may not be importable in tests
    from .. import runtime as _runtime_module
except Exception:
    _runtime_module = None  # type: ignore

try:  # FastAPI >=0.99 exposes Request at top-level
    from fastapi import Request  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - allow older FastAPI versions
    "resolve_router_context",
    try:
        from starlette.requests import Request  # type: ignore
    except ImportError:  # pragma: no cover - minimal fallback for tests
        class Request:  # type: ignore
            headers: Any  # type: ignore
_TRUTHY = {"1", "true", "yes", "on"}
_SUPPORTED_HMAC: Dict[str, Callable[[bytes], Any]] = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


_API_KEY = os.getenv("NAMEL3SS_API_KEY")
_CUSTOM_DEPENDENCY_PATH = os.getenv("NAMEL3SS_ROUTER_DEPENDENCY")
_JWT_SECRET = os.getenv("NAMEL3SS_JWT_SECRET")
_JWT_ALGORITHMS = [
    alg.strip().upper()
    for alg in (os.getenv("NAMEL3SS_JWT_ALGORITHMS") or "HS256").split(",")
    if alg.strip()
]
_JWT_ALGORITHMS = [alg for alg in _JWT_ALGORITHMS if alg in _SUPPORTED_HMAC]
if not _JWT_ALGORITHMS:
    _JWT_ALGORITHMS = ["HS256"]
_JWT_AUDIENCES = [
    value.strip()
    for value in (os.getenv("NAMEL3SS_JWT_AUDIENCE") or "").split(",")
    if value.strip()
]
_JWT_ISSUERS = [
    value.strip()
    for value in (os.getenv("NAMEL3SS_JWT_ISSUER") or "").split(",")
    if value.strip()
]
try:
    _JWT_LEEWAY = int(os.getenv("NAMEL3SS_JWT_LEEWAY", "0"))
except ValueError:
    _JWT_LEEWAY = 0

_AUTH_MODE = (os.getenv("NAMEL3SS_AUTH_MODE") or "").strip().lower()
_ENABLE_TENANT_RESOLUTION = _is_truthy(os.getenv("NAMEL3SS_ENABLE_TENANT_RESOLUTION"))
_TENANT_HEADER = os.getenv("NAMEL3SS_TENANT_HEADER") or "x-tenant-id"
_TENANT_CLAIMS = [
    claim.strip()
    for claim in (os.getenv("NAMEL3SS_TENANT_CLAIM") or "tenant,tenant_id,tid").split(",")
    if claim.strip()
]
_TENANT_REQUIRED = _is_truthy(os.getenv("NAMEL3SS_REQUIRE_TENANT"))
_ALLOW_CUSTOM_ANONYMOUS = _is_truthy(os.getenv("NAMEL3SS_ALLOW_ANONYMOUS"))

_CUSTOM_DEPENDENCY: Optional[Callable[..., Any]] = None

if _AUTH_MODE not in {"", "disabled", "optional", "required"}:
    _LOGGER.warning("Unknown NAMEL3SS_AUTH_MODE '%s'; defaulting to 'disabled'", _AUTH_MODE)
    _AUTH_MODE = "disabled"

if not _AUTH_MODE:
    if _JWT_SECRET:
        _AUTH_MODE = "required"
    elif _ENABLE_TENANT_RESOLUTION or _TENANT_REQUIRED:
        _AUTH_MODE = "optional"
    else:
        _AUTH_MODE = "disabled"

if _AUTH_MODE == "required" and not _JWT_SECRET:
    _LOGGER.warning(
        "Auth mode 'required' configured without NAMEL3SS_JWT_SECRET; disabling JWT verification."
    )
    _AUTH_MODE = "disabled"

_AUTH_CONTEXT_ENABLED = _AUTH_MODE in {"optional", "required"} or _TENANT_REQUIRED
_AUTH_REQUIRED = _AUTH_MODE == "required"
_ALLOW_ANONYMOUS = (_AUTH_MODE == "optional") or _ALLOW_CUSTOM_ANONYMOUS

if _TENANT_REQUIRED and not (_ENABLE_TENANT_RESOLUTION or _JWT_SECRET):
    _ENABLE_TENANT_RESOLUTION = True


class JWTValidationError(Exception):
    """Raised when a JWT fails validation."""


def _urlsafe_b64decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _decode_hmac_jwt(token: str) -> Dict[str, Any]:
    if not _JWT_SECRET:
        raise JWTValidationError("JWT secret is not configured")
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise JWTValidationError("Malformed JWT structure") from exc
    try:
        header = json.loads(_urlsafe_b64decode(header_segment))
        payload = json.loads(_urlsafe_b64decode(payload_segment))
    except Exception as exc:  # pragma: no cover - defensive
        raise JWTValidationError("Invalid JWT encoding") from exc
    algorithm = str(header.get("alg") or _JWT_ALGORITHMS[0]).upper()
    if algorithm not in _JWT_ALGORITHMS or algorithm not in _SUPPORTED_HMAC:
        raise JWTValidationError(f"Unsupported JWT algorithm '{algorithm}'")
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected = hmac.new(_JWT_SECRET.encode("utf-8"), signing_input, _SUPPORTED_HMAC[algorithm]).digest()
    provided = _urlsafe_b64decode(signature_segment)
    if not hmac.compare_digest(expected, provided):
        raise JWTValidationError("Invalid token signature")
    return payload


def _validate_claims(claims: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    exp = claims.get("exp")
    if exp is not None:
        try:
            if float(exp) + _JWT_LEEWAY < now:
                raise JWTValidationError("Token has expired")
        except (TypeError, ValueError) as exc:
            raise JWTValidationError("Invalid 'exp' claim") from exc
    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            if float(nbf) - _JWT_LEEWAY > now:
                raise JWTValidationError("Token is not yet valid")
        except (TypeError, ValueError) as exc:
            raise JWTValidationError("Invalid 'nbf' claim") from exc
    if _JWT_ISSUERS:
        issuer = claims.get("iss")
        if issuer not in _JWT_ISSUERS:
            raise JWTValidationError("Token issuer is not permitted")
    if _JWT_AUDIENCES:
        audience = claims.get("aud")
        if isinstance(audience, str):
            audiences = [audience]
        elif isinstance(audience, (list, tuple, set)):
            audiences = [str(item) for item in audience if str(item).strip()]
        else:
            audiences = []
        if not any(value in audiences for value in _JWT_AUDIENCES):
            raise JWTValidationError("Token audience is not permitted")
    return claims


def _verify_jwt_token(token: str) -> Dict[str, Any]:
    return _validate_claims(_decode_hmac_jwt(token))

__all__ = [
    "GENERATED_ROUTERS",
    "include_generated_routers",
    "router_dependencies",
]


_LOGGER = logging.getLogger("namel3ss.security")
_API_KEY = os.getenv("NAMEL3SS_API_KEY")
_CUSTOM_DEPENDENCY_PATH = os.getenv("NAMEL3SS_ROUTER_DEPENDENCY")
_CUSTOM_DEPENDENCY: Optional[Callable[..., Any]] = None


def _load_custom_dependency() -> Optional[Callable[..., Any]]:
    path = _CUSTOM_DEPENDENCY_PATH
    if not path:
        return None
    candidate_path = path.strip()
    if not candidate_path:
        return None
    module_name: Optional[str]
    attr_name: Optional[str]
    if ":" in candidate_path:
        module_name, attr_name = candidate_path.rsplit(":", 1)
    else:
        module_name, attr_name = candidate_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        candidate = getattr(module, attr_name)
    except Exception:
        _LOGGER.exception("Failed to import router dependency '%s'", candidate_path)
        return None
    if not callable(candidate):
        _LOGGER.warning("Router dependency '%s' is not callable", candidate_path)
        return None
    return candidate


def _make_http_exception(status_code: int, detail: Any) -> HTTPException:
    try:
        exc = HTTPException(status_code=status_code, detail=detail)
    except TypeError:  # pragma: no cover - fallback for minimal stubs
        exc = HTTPException(status_code, detail)
    if not hasattr(exc, "status_code"):
        setattr(exc, "status_code", status_code)
    if detail is not None and not hasattr(exc, "detail"):
        setattr(exc, "detail", detail)
    return exc


def _clean_authorization(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def _extract_bearer_token(request: Request) -> Optional[str]:
    headers = getattr(request, "headers", {}) or {}
    auth_value = headers.get("authorization") if hasattr(headers, "get") else None
    if auth_value is None and hasattr(headers, "get"):
        auth_value = headers.get("Authorization")
    if isinstance(auth_value, str):
        return _clean_authorization(auth_value)
    return None


def _normalize_scopes(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items: List[str] = []
        for part in raw.replace(",", " ").split():
            part = part.strip()
            if part:
                items.append(part)
        return items
    if isinstance(raw, (list, tuple, set)):
        items = []
        for value in raw:
            text = str(value).strip()
            if text:
                items.append(text)
        return items
    return []


def _resolve_tenant(claims: Optional[Dict[str, Any]], request: Request) -> Optional[str]:
    if _ENABLE_TENANT_RESOLUTION or _TENANT_REQUIRED:
        headers = getattr(request, "headers", {}) or {}
        candidate = None
        if hasattr(headers, "get"):
            candidate = headers.get(_TENANT_HEADER)
            if candidate is None:
                candidate = headers.get(_TENANT_HEADER.lower())
            if candidate is None:
                candidate = headers.get(_TENANT_HEADER.upper())
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if not claims:
        return None
    for claim_name in _TENANT_CLAIMS:
        value = claims.get(claim_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_request_context(claims: Optional[Dict[str, Any]], request: Request) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    if claims:
        context["claims"] = dict(claims)
        subject = claims.get("sub")
        if isinstance(subject, str) and subject.strip():
            context["subject"] = subject.strip()
        scopes = _normalize_scopes(claims.get("scope") or claims.get("scopes"))
        if scopes:
            context["scopes"] = scopes
    tenant_value = _resolve_tenant(claims, request)
    if tenant_value is not None:
        context["tenant"] = tenant_value
    return context


def _store_request_context(request: Request, context: Dict[str, Any]) -> None:
    try:
        state = getattr(request, "state", None)
        if state is not None:
            setattr(state, "namel3ss", context)
    except Exception:  # pragma: no cover - defensive
        _LOGGER.debug("Unable to attach context to request.state")
    if _runtime_module is not None:
        setter = getattr(_runtime_module, "set_request_context", None)
        clearer = getattr(_runtime_module, "clear_request_context", None)
        try:
            if callable(setter):
                setter(context)
            elif not context and callable(clearer):
                clearer()
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception("Failed to propagate request context to runtime")


def resolve_router_context() -> Dict[str, Any]:
    if _runtime_module is None:
        return {}
    getter = getattr(_runtime_module, "get_request_context", None)
    if callable(getter):
        try:
            return getter({})
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception("Failed to read runtime request context")
    return {}


async def _require_api_key(request: Request) -> None:
    if not _API_KEY:
        return
    headers = request.headers
    provided = headers.get("x-api-key") or _clean_authorization(headers.get("authorization"))
    if provided != _API_KEY:
        _LOGGER.warning("Rejected request with invalid API key header")
        raise _make_http_exception(401, "Invalid API key")


async def _require_auth_context(request: Request) -> Dict[str, Any]:
    if not _AUTH_CONTEXT_ENABLED:
        _store_request_context(request, {})
        return {}

    token = _extract_bearer_token(request)
    claims: Optional[Dict[str, Any]] = None

    if token:
        try:
            claims = _verify_jwt_token(token)
        except JWTValidationError as exc:
            _store_request_context(request, {})
            _LOGGER.warning("Rejected request with invalid JWT: %s", exc)
            raise _make_http_exception(401, str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            _store_request_context(request, {})
            _LOGGER.exception("Unexpected JWT validation failure")
            raise _make_http_exception(401, "Invalid bearer token") from exc
    elif _AUTH_REQUIRED and not _ALLOW_ANONYMOUS:
        _store_request_context(request, {})
        raise _make_http_exception(401, "Missing bearer token")

    context = _build_request_context(claims, request)
    if not context and claims:
        context = {"claims": dict(claims)}
    _store_request_context(request, context)

    if _TENANT_REQUIRED and context.get("tenant") is None:
        raise _make_http_exception(403, "Tenant is required")

    return context


class _DependencyWrapper:
    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.dependency = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


def _make_dependency(func: Callable[..., Any]) -> Any:
    """Ensure dependency objects expose `.dependency` even when FastAPI is stubbed."""
    candidate = Depends(func)
    if not hasattr(candidate, "dependency"):
        try:
            setattr(candidate, "dependency", func)
            return candidate
        except Exception:  # pragma: no cover - defensive fallback
            return _DependencyWrapper(func)
    return candidate


def router_dependencies() -> List[Any]:
    """Return dependencies applied to every generated router."""

    global _CUSTOM_DEPENDENCY
    dependencies: List[Any] = []
    if _AUTH_CONTEXT_ENABLED:
        dependencies.append(_make_dependency(_require_auth_context))
    if _CUSTOM_DEPENDENCY_PATH and _CUSTOM_DEPENDENCY is None:
        _CUSTOM_DEPENDENCY = _load_custom_dependency()
    if _CUSTOM_DEPENDENCY is not None:
        dependencies.append(_make_dependency(_CUSTOM_DEPENDENCY))
    if _API_KEY:
        dependencies.append(_make_dependency(_require_api_key))
    return dependencies


def _generated_routers() -> Iterable:
    from ..routers import GENERATED_ROUTERS  # local import to avoid circular dependency

    return GENERATED_ROUTERS


class _GeneratedRoutersProxy:
    def __iter__(self):
        return iter(_generated_routers())

    def __len__(self) -> int:
        return len(list(_generated_routers()))

    def __getitem__(self, index: int) -> Any:
        return list(_generated_routers())[index]


GENERATED_ROUTERS = _GeneratedRoutersProxy()


def include_generated_routers(app: FastAPI, routers: Optional[Iterable] = None) -> None:
    """Attach generated routers to ``app`` while allowing custom overrides."""

    target = routers or _generated_routers()
    for router in target:
        app.include_router(router)
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_custom_readme() -> str:
    template = '''
# Custom Backend Extensions

This folder is reserved for your handcrafted FastAPI routes and helpers. The
code generator creates it once and will not overwrite files you add here.

- Put reusable dependencies in `__init__.py` or new modules.
- Add route overrides in `routes/custom_api.py` and register them on the
  module-level `router` instance.
- Use the optional `setup(app)` hook to run initialization logic after the
  generated routers are attached (for example, authentication, middleware, or
  event handlers).

Whenever you run the Namel3ss generator again your custom code stays intact.
Refer to the generated modules under `generated/` for available helpers.
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_custom_api_stub() -> str:
    template = '''
"""Custom API extensions for your Namel3ss backend.

This module is created once. Add routes, dependencies, or hooks freely; the
generator will never overwrite it.
"""

from __future__ import annotations

from fastapi import APIRouter

# Optional helpers when you want to reuse generated logic:
# from ..generated.routers import experiments as generated_experiments
# from ..generated.routers import models as generated_models
# from ..generated.schemas import ExperimentResult, PredictionResponse

router = APIRouter()


# Example override (uncomment and adapt):
# @router.post(
#     "/api/models/{model_name}/predict",
#     response_model=PredictionResponse,
#     include_in_schema=False,
# )
# async def predict_with_tracking(model_name: str, payload: dict) -> PredictionResponse:
#     base = await generated_models.predict(model_name, payload)
#     base.metadata.setdefault("tags", []).append("customized")
#     base.metadata["handled_by"] = "custom_api"
#     return base
#
# Example extension:
# @router.get("/api/experiments/{slug}/summary", response_model=ExperimentResult)
# async def experiment_summary(slug: str) -> ExperimentResult:
#     result = await generated_experiments.get_experiment(slug)
#     result.metadata["summary"] = f"Experiment {slug} served by custom routes."
#     return result
#
# The optional ``setup`` hook runs after generated routers are registered.


def setup(app) -> None:  # pragma: no cover - user may replace implementation
    """Run initialization after generated routers are registered."""

    _ = app  # Replace with custom logic (auth, logging, etc.)
'''
    return textwrap.dedent(template).strip() + "\n"
