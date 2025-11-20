"""HTTP client functionality for LLM API calls."""

from textwrap import dedent

HTTP_CLIENT = dedent(
    '''
def _http_post_json(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> tuple[int, str, Optional[Any]]:
    import json as _json
    import urllib.request
    import random

    request_headers = {
        str(key): str(value)
        for key, value in headers.items()
    }
    request_headers.setdefault("Content-Type", "application/json")

    try:
        import httpx as _httpx  # type: ignore
    except Exception:  # pragma: no cover - optional dependency guard
        _httpx = None  # type: ignore

    if _httpx is not None:
        # Get connector config from runtime settings for retry behavior
        connector_settings = RUNTIME_SETTINGS.get("connectors", {})
        retry_max_attempts = connector_settings.get("retry_max_attempts", 3)
        retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
        retry_max_delay = connector_settings.get("retry_max_delay", 5.0)
        
        client_kwargs: Dict[str, Any] = {}
        try:
            client_kwargs["timeout"] = _httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
            
        last_error: Optional[Exception] = None
        for attempt in range(1, retry_max_attempts + 1):
            try:
                with _httpx.Client(**client_kwargs) as client:
                    response = client.request(
                        "POST",
                        url,
                        json=data,
                        headers=request_headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    status_code = int(getattr(response, "status_code", 0))
                    parsed: Optional[Any]
                    try:
                        parsed = response.json()
                    except Exception:
                        parsed = None
                    text = ""
                    raw_text_candidate = getattr(response, "text", "")
                    if isinstance(raw_text_candidate, str) and raw_text_candidate:
                        text = raw_text_candidate
                    elif parsed is not None:
                        try:
                            text = _json.dumps(parsed)
                        except Exception:
                            text = ""
                    else:
                        raw_bytes = getattr(response, "content", b"")
                        if isinstance(raw_bytes, (bytes, bytearray)):
                            text = bytes(raw_bytes).decode("utf-8", "replace")
                    return status_code, text, parsed
            except Exception as exc:
                last_error = exc
                if attempt < retry_max_attempts:
                    # Calculate exponential backoff with jitter
                    delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    import time
                    time.sleep(delay + jitter)
                else:
                    raise last_error from exc

    payload_bytes = _json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload_bytes, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = response.getcode()
        raw_bytes = response.read()

    text = raw_bytes.decode("utf-8", "replace")
    try:
        parsed = _json.loads(text)
    except Exception:
        parsed = None
    return int(status_code or 0), text, parsed
'''
).strip()

__all__ = ['HTTP_CLIENT']
