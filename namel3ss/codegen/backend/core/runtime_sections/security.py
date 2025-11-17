from __future__ import annotations

from textwrap import dedent

SECURITY_SECTION = dedent(
    '''

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_SENSITIVE_HEADERS = {
    "server",
    "x-powered-by",
}
_CSRF_COOKIE_NAME = os.getenv("NAMEL3SS_CSRF_COOKIE", "namel3ss-csrf")
_CSRF_HEADER_NAME = os.getenv("NAMEL3SS_CSRF_HEADER", "x-csrf-token")
_CSRF_COOKIE_PATH = os.getenv("NAMEL3SS_CSRF_COOKIE_PATH", "/")
_CSRF_COOKIE_SAMESITE = (os.getenv("NAMEL3SS_CSRF_COOKIE_SAMESITE") or "lax").title()
try:
    _CSRF_COOKIE_MAX_AGE = int(os.getenv("NAMEL3SS_CSRF_TTL", "43200"))
except ValueError:
    _CSRF_COOKIE_MAX_AGE = 43200
_CSRF_SECRET_ENV = os.getenv("NAMEL3SS_CSRF_SECRET") or os.getenv("NAMEL3SS_SECRET_KEY")
_CSRF_SECRET = None
_CSRF_SECRET_BYTES: bytes
_RATE_LIMITS = {
    "auth": os.getenv("NAMEL3SS_RATE_LIMIT_AUTH"),
    "ai": os.getenv("NAMEL3SS_RATE_LIMIT_AI"),
    "experiments": os.getenv("NAMEL3SS_RATE_LIMIT_EXPERIMENTS"),
}
_RATE_LIMIT_DEFAULTS = {
    "auth": (10, 60),
    "ai": (60, 60),
    "experiments": (30, 60),
}
_RATE_LIMIT_BUCKETS: Dict[str, Dict[str, int]] = {}
_RATE_LIMIT_STATE: Dict[str, deque] = defaultdict(deque)
_RATE_LIMIT_LOCK = threading.Lock()


def _is_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY_VALUES


def get_env_setting(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def is_production_mode() -> bool:
    runtime_value = _runtime_setting("environment")
    if runtime_value:
        candidate = str(runtime_value).strip().lower()
    else:
        candidate = str(os.getenv("NAMEL3SS_ENV") or os.getenv("ENV") or "development").strip().lower()
    return candidate in {"prod", "production"}


def is_debug_mode() -> bool:
    runtime_value = _runtime_setting("debug")
    if runtime_value is not None:
        return bool(runtime_value)
    env_value = os.getenv("NAMEL3SS_DEBUG")
    if env_value is not None:
        return _is_truthy(env_value)
    return not is_production_mode()


def get_secret(name: str, default: Optional[str] = None, *, allow_generate: bool = False) -> str:
    candidate = os.getenv(name)
    if candidate:
        return candidate
    runtime_secrets = _runtime_setting("secrets")
    if isinstance(runtime_secrets, dict) and runtime_secrets.get(name):
        return str(runtime_secrets[name])
    if allow_generate and not is_production_mode():
        generated = secrets.token_urlsafe(48)
        logger.warning("Secret %s was generated automatically for development.", name)
        return generated
    return default or ""


_CSRF_SECRET = _CSRF_SECRET_ENV or get_secret("NAMEL3SS_CSRF_SECRET", allow_generate=True)
_CSRF_SECRET_BYTES = (_CSRF_SECRET or "").encode("utf-8") if _CSRF_SECRET else secrets.token_bytes(32)


def csrf_protection_enabled() -> bool:
    runtime_value = _runtime_setting("csrf_enabled")
    if runtime_value is not None:
        return bool(runtime_value)
    env_value = os.getenv("NAMEL3SS_ENABLE_CSRF")
    if env_value is not None:
        return _is_truthy(env_value)
    return True


def csrf_cookie_name() -> str:
    return _CSRF_COOKIE_NAME


def csrf_header_name() -> str:
    return _CSRF_HEADER_NAME


def _sign_csrf(token: str) -> str:
    return hmac.new(_CSRF_SECRET_BYTES, token.encode("utf-8"), hashlib.sha256).hexdigest()


def _serialize_csrf_token(token: str) -> str:
    return f"{token}:{_sign_csrf(token)}"


def _deserialize_csrf_token(payload: Optional[str]) -> Optional[str]:
    if not payload or ":" not in payload:
        return None
    token, signature = payload.split(":", 1)
    expected = _sign_csrf(token)
    if hmac.compare_digest(expected, signature):
        return token
    return None


def ensure_csrf_cookie(cookies: Optional[Dict[str, Any]]) -> Tuple[Optional[str], bool]:
    if not csrf_protection_enabled():
        return None, False
    cookie_value = None
    if cookies and hasattr(cookies, "get"):
        cookie_value = cookies.get(_CSRF_COOKIE_NAME)
    token = _deserialize_csrf_token(cookie_value)
    if token:
        return cookie_value, False
    token = secrets.token_urlsafe(32)
    return _serialize_csrf_token(token), True


def should_enforce_csrf(method: str) -> bool:
    if not csrf_protection_enabled():
        return False
    return method.upper() in {"POST", "PUT", "PATCH", "DELETE"}


def validate_csrf_request(method: str, headers: Optional[Dict[str, Any]], cookies: Optional[Dict[str, Any]]) -> bool:
    if not should_enforce_csrf(method):
        return True
    cookie_value = None
    if cookies and hasattr(cookies, "get"):
        cookie_value = cookies.get(_CSRF_COOKIE_NAME)
    header_value = None
    if headers and hasattr(headers, "get"):
        header_value = headers.get(_CSRF_HEADER_NAME) or headers.get(_CSRF_HEADER_NAME.lower())
    token = _deserialize_csrf_token(cookie_value)
    if not token or not header_value:
        return False
    return hmac.compare_digest(token, header_value.strip())


def csrf_cookie_settings() -> Dict[str, Any]:
    return {
        "path": _CSRF_COOKIE_PATH,
        "httponly": False,
        "secure": _is_truthy(os.getenv("NAMEL3SS_CSRF_COOKIE_SECURE"), default=is_production_mode()),
        "samesite": _CSRF_COOKIE_SAMESITE,
        "max_age": _CSRF_COOKIE_MAX_AGE,
    }


def set_csrf_cookie(response: Any, value: str) -> None:
    if not value or not csrf_protection_enabled():
        return
    setter = getattr(response, "set_cookie", None)
    if not callable(setter):
        return
    params = csrf_cookie_settings()
    try:
        setter(
            _CSRF_COOKIE_NAME,
            value,
            max_age=params["max_age"],
            path=params["path"],
            secure=params["secure"],
            httponly=params["httponly"],
            samesite=params["samesite"],
        )
    except Exception:
        logger.debug("Unable to attach CSRF cookie")


def apply_security_headers(response: Any, request_id: Optional[str] = None) -> None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return

    def _set_header(key: str, value: str) -> None:
        try:
            if hasattr(headers, "get"):
                current = headers.get(key)  # type: ignore[attr-defined]
                if current is None:
                    headers[key] = value  # type: ignore[index]
            elif key not in headers:
                headers[key] = value  # type: ignore[index]
        except Exception:
            headers[key] = value  # type: ignore[index]

    try:
        _set_header("x-frame-options", "DENY")
        _set_header("x-content-type-options", "nosniff")
        _set_header("referrer-policy", "same-origin")
        _set_header("x-xss-protection", "1; mode=block")
        csp = "default-src 'self'; frame-ancestors 'none'; img-src 'self' data:;"
        _set_header("content-security-policy", csp)
        if request_id:
            _set_header("x-request-id", request_id)
        for header in _SENSITIVE_HEADERS:
            if header in headers:
                headers[header] = "redacted"
    except Exception:
        logger.debug("Unable to apply security headers")


def _parse_rate_limit(value: Optional[str], bucket: str) -> Dict[str, int]:
    default_limit, default_window = _RATE_LIMIT_DEFAULTS[bucket]
    if not value:
        return {"limit": default_limit, "window": default_window}
    text = value.strip().lower()
    if "/" in text:
        limit_part, window_part = text.split("/", 1)
    else:
        limit_part, window_part = text, "60s"
    try:
        limit = int(limit_part)
    except ValueError:
        limit = default_limit
    multiplier = 1
    if window_part.endswith("min"):
        multiplier = 60
        window_part = window_part[:-3]
    elif window_part.endswith("m"):
        multiplier = 60
        window_part = window_part[:-1]
    elif window_part.endswith("h"):
        multiplier = 3600
        window_part = window_part[:-1]
    elif window_part.endswith("s"):
        window_part = window_part[:-1]
    try:
        window = int(float(window_part) * multiplier)
    except ValueError:
        window = default_window
    window = max(window, 1)
    limit = max(limit, 1)
    return {"limit": limit, "window": window}


for bucket, value in _RATE_LIMITS.items():
    _RATE_LIMIT_BUCKETS[bucket] = _parse_rate_limit(value, bucket)


class RateLimitExceeded(Exception):
    def __init__(self, bucket: str, limit: int, window: int) -> None:
        super().__init__(f"Rate limit exceeded for {bucket}")
        self.bucket = bucket
        self.limit = limit
        self.window = window


def describe_rate_limit(bucket: str) -> Dict[str, int]:
    return dict(_RATE_LIMIT_BUCKETS.get(bucket, {}))


def enforce_rate_limit(bucket: str, identifier: str) -> None:
    config = _RATE_LIMIT_BUCKETS.get(bucket)
    if not config:
        return
    limit = int(config.get("limit", 0))
    window = int(config.get("window", 0))
    if limit <= 0 or window <= 0:
        return
    key = f"{bucket}:{identifier or 'anonymous'}"
    now = time.time()
    with _RATE_LIMIT_LOCK:
        entries = _RATE_LIMIT_STATE[key]
        while entries and entries[0] <= now - window:
            entries.popleft()
        if len(entries) >= limit:
            raise RateLimitExceeded(bucket, limit, window)
        entries.append(now)

'''
).strip()

__all__ = ["SECURITY_SECTION"]
