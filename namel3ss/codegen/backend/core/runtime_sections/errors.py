from __future__ import annotations

from textwrap import dedent

ERRORS_SECTION = dedent(
    '''

class Namel3ssError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


def _safe_message(exc: Exception) -> str:
    text = str(exc) if exc else ""
    return text or exc.__class__.__name__


def _request_id_from_headers(headers: Optional[Dict[str, Any]]) -> Optional[str]:
    if not headers or not hasattr(headers, "get"):
        return None
    candidate = headers.get("x-request-id") or headers.get("X-Request-ID")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def ensure_request_id(headers: Optional[Dict[str, Any]] = None) -> str:
    header_value = _request_id_from_headers(headers)
    if header_value:
        return header_value
    return uuid.uuid4().hex


def build_error_payload(
    code: str,
    message: str,
    *,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    payload = {
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    if request_id:
        payload["request_id"] = request_id
    return status_code, payload


def format_error_response(exc: Exception, request_id: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    status_code = getattr(exc, "status_code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    if isinstance(exc, Namel3ssError):
        message = _safe_message(exc)
        return build_error_payload(exc.code, message, status_code=status_code, details=exc.details, request_id=request_id)
    if hasattr(exc, "detail"):
        detail = getattr(exc, "detail")
        detail_message = detail if isinstance(detail, str) else _safe_message(exc)
        return build_error_payload("http_error", detail_message, status_code=status_code, request_id=request_id)
    if is_debug_mode():
        logger.exception("Unhandled exception")
        return build_error_payload("internal_server_error", _safe_message(exc), status_code=500, request_id=request_id)
    logger.exception("Unhandled exception")
    return build_error_payload("internal_server_error", "An unexpected error occurred.", status_code=500, request_id=request_id)


def apply_error_response(response: Any, status_code: int, payload: Dict[str, Any]) -> Any:
    setter = getattr(response, "status_code", None)
    if isinstance(setter, int):
        response.status_code = status_code
    headers = getattr(response, "headers", None)
    if headers is not None:
        headers.setdefault("content-type", "application/json")
    body_setter = getattr(response, "body", None)
    if body_setter is not None:
        response.body = json.dumps(payload).encode("utf-8")
    return response

'''
).strip()

__all__ = ["ERRORS_SECTION"]
