from __future__ import annotations

from textwrap import dedent

OBSERVABILITY_SECTION = dedent(
    '''
START_TIME = time.time()
_LOGGING_CONFIGURED = False
_TRACING_CONFIGURED = False
_METRIC_LOCK = threading.RLock()
_REQUEST_METRICS: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "errors": 0,
})
_DATASET_METRICS: Dict[str, Dict[str, float]] = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0.0,
    "rows": 0.0,
})
_CONNECTOR_STATUS: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


def _iso_timestamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "namel3ss_event"):
            payload["event"] = record.namel3ss_event  # type: ignore[attr-defined]
        if hasattr(record, "namel3ss_data"):
            payload["data"] = record.namel3ss_data  # type: ignore[attr-defined]
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleLogFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("[%(levelname)s] %(asctime)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras: Dict[str, Any] = {}
        if hasattr(record, "namel3ss_event"):
            extras["event"] = record.namel3ss_event  # type: ignore[attr-defined]
        if hasattr(record, "namel3ss_data"):
            extras["data"] = record.namel3ss_data  # type: ignore[attr-defined]
        if extras:
            return f"{base} | {json.dumps(extras, default=str)}"
        return base


def configure_logging(level: Optional[str] = None) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not level:
        return
    log_level = (level or os.getenv("NAMEL3SS_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    handler = logging.StreamHandler()
    formatter: logging.Formatter
    if is_production_mode():
        formatter = _JsonLogFormatter()
    else:
        formatter = _ConsoleLogFormatter()
    handler.setFormatter(formatter)
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level, logging.INFO))
    _LOGGING_CONFIGURED = True


def request_timer() -> float:
    return time.perf_counter()


def _normalize_route(path: str) -> str:
    if not path:
        return "/"
    normalized = path.split("?", 1)[0]
    return normalized or "/"


def _update_request_metrics(route: str, method: str, status_code: int, duration_ms: float) -> None:
    with _METRIC_LOCK:
        stats = _REQUEST_METRICS[(route, method)]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["max_ms"] = max(duration_ms, stats["max_ms"])
        if status_code >= 500:
            stats["errors"] += 1


def record_request_observation(
    started_at: float,
    path: str,
    method: str,
    status_code: int,
    request_id: Optional[str] = None,
    client_host: Optional[str] = None,
) -> Dict[str, Any]:
    duration_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
    route = _normalize_route(path)
    verb = (method or "GET").upper()
    _update_request_metrics(route, verb, status_code, duration_ms)
    payload = {
        "route": route,
        "method": verb,
        "status": status_code,
        "duration_ms": round(duration_ms, 3),
    }
    if request_id:
        payload["request_id"] = request_id
    if client_host:
        payload["client_ip"] = client_host
    logger.info(
        "HTTP request",
        extra={
            "namel3ss_event": "http_request",
            "namel3ss_data": payload,
        },
    )
    return payload


def observe_dataset_stage(dataset: Optional[str], stage: str, duration_ms: float, rows: int) -> None:
    if not dataset:
        return
    with _METRIC_LOCK:
        stats = _DATASET_METRICS[dataset]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["rows"] += rows


def observe_dataset_fetch(dataset: Optional[str], status: str, duration_ms: float, rows: int, cache_state: str) -> None:
    if not dataset:
        return
    with _METRIC_LOCK:
        stats = _DATASET_METRICS[dataset]
        stats.setdefault("status", status)
        stats.setdefault("cache", cache_state)
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["rows"] += rows


def observe_connector_status(name: Optional[str], status: str) -> None:
    key = name or "unnamed"
    with _METRIC_LOCK:
        _CONNECTOR_STATUS[key][status] += 1


def _format_labels(labels: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(labels):
        value = str(labels[key]).replace('"', '\\"')
        parts.append(f'{key}="{value}"')
    return ",".join(parts)


def render_prometheus_metrics() -> str:
    lines: List[str] = []
    lines.append("# HELP namel3ss_uptime_seconds Application uptime")
    lines.append("# TYPE namel3ss_uptime_seconds gauge")
    lines.append(f"namel3ss_uptime_seconds {max(time.time() - START_TIME, 0.0):.3f}")
    with _METRIC_LOCK:
        if _REQUEST_METRICS:
            lines.append("# HELP namel3ss_request_count Total requests per route")
            lines.append("# TYPE namel3ss_request_count counter")
            for (route, method), stats in sorted(_REQUEST_METRICS.items()):
                labels = _format_labels({"route": route, "method": method})
                lines.append(f"namel3ss_request_count{{{labels}}} {int(stats['count'])}")
            lines.append("# HELP namel3ss_request_duration_seconds_total Total request duration")
            lines.append("# TYPE namel3ss_request_duration_seconds_total counter")
            for (route, method), stats in sorted(_REQUEST_METRICS.items()):
                labels = _format_labels({"route": route, "method": method})
                seconds = stats['total_ms'] / 1000.0
                lines.append(f"namel3ss_request_duration_seconds_total{{{labels}}} {seconds:.6f}")
        if _DATASET_METRICS:
            lines.append("# HELP namel3ss_dataset_fetch_seconds Dataset fetch runtime")
            lines.append("# TYPE namel3ss_dataset_fetch_seconds counter")
            for dataset, stats in sorted(_DATASET_METRICS.items()):
                labels = _format_labels({"dataset": dataset, "cache": stats.get("cache", "unknown")})
                seconds = stats['total_ms'] / 1000.0
                lines.append(f"namel3ss_dataset_fetch_seconds{{{labels}}} {seconds:.6f}")
                lines.append(f"namel3ss_dataset_rows_total{{{labels}}} {int(stats['rows'])}")
        if _CONNECTOR_STATUS:
            lines.append("# HELP namel3ss_connector_status Connector executions grouped by status")
            lines.append("# TYPE namel3ss_connector_status counter")
            for name, statuses in sorted(_CONNECTOR_STATUS.items()):
                for status, count in sorted(statuses.items()):
                    labels = _format_labels({"connector": name, "status": status})
                    lines.append(f"namel3ss_connector_status{{{labels}}} {int(count)}")
    lines.append("")
    return "\\n".join(lines)


def health_summary() -> Dict[str, Any]:
    uptime = max(time.time() - START_TIME, 0.0)
    payload = {
        "status": "ok",
        "uptime_seconds": round(uptime, 3),
    }
    if APP:
        payload["app"] = {
            "name": APP.get("name"),
            "version": APP.get("version"),
        }
    return payload


async def _database_ready(timeout: float = 2.0) -> Tuple[bool, str]:
    try:
        from .. import database as _generated_database  # type: ignore
    except Exception:
        return True, "skipped"
    engine = getattr(_generated_database, "engine", None)
    if engine is None:
        return False, "engine_missing"
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        logger.warning("Database readiness check failed: %s", exc)
        return False, str(exc)


async def readiness_checks() -> Dict[str, Any]:
    database_ready, detail = await _database_ready()
    status = "ok" if database_ready else "error"
    return {
        "status": status,
        "checks": {
            "database": {
                "ok": database_ready,
                "detail": detail,
            }
        },
    }


def configure_tracing(app: Optional[Any] = None) -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return
    if os.getenv("NAMEL3SS_ENABLE_TRACING", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception:
        logger.debug("OpenTelemetry SDK not installed; tracing disabled")
        return
    service_name = APP.get("name") if isinstance(APP, dict) else "namel3ss"
    provider = TracerProvider(resource=Resource.create({"service.name": service_name or "namel3ss"}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
        except Exception:
            logger.debug("FastAPI instrumentation unavailable for tracing")
    _TRACING_CONFIGURED = True
'''
).strip()

__all__ = ["OBSERVABILITY_SECTION"]
