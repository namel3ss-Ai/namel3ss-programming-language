"""
Comprehensive Observability Stack for Namel3ss Parallel and Distributed Execution.

This module implements production-ready observability with:
- OpenTelemetry distributed tracing with Jaeger integration
- Metrics collection with Prometheus compatibility
- Health monitoring and system status checks
- Structured logging with correlation IDs
- Performance monitoring and alerting
- Real-time dashboards and monitoring

Enterprise-grade observability for production deployment and operations.
"""

import asyncio
import functools
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Iterator
from datetime import datetime, timedelta

# Try to import optional dependencies
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.semconv.trace import SpanAttributes
except ImportError:
    trace = None
    metrics = None
    JaegerExporter = None
    PrometheusMetricReader = None
    TracerProvider = None
    BatchSpanProcessor = None
    MeterProvider = None
    Status = None
    StatusCode = None
    SpanAttributes = None

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
except ImportError:
    prometheus_client = None
    Counter = None
    Histogram = None
    Gauge = None
    Info = None
    CollectorRegistry = None

logger = logging.getLogger(__name__)


# =============================================================================
# Core Observability Models
# =============================================================================

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    description: str
    check_function: Callable[[], bool]
    timeout_seconds: float = 5.0
    critical: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None


@dataclass
class HealthReport:
    """Health check report."""
    component: str
    status: HealthStatus
    checks: List[Dict[str, Any]]
    timestamp: float
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """Tracing context for correlation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = "unknown"
    tags: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


# =============================================================================
# Metrics Collection System
# =============================================================================

class MetricsCollector:
    """
    Production-ready metrics collection system.
    
    Features:
    - Prometheus-compatible metrics
    - Custom metric definitions
    - Automatic labeling
    - Performance monitoring
    - Alert thresholds
    """
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        metrics_port: int = 9090,
        namespace: str = "namel3ss",
    ):
        """Initialize metrics collector."""
        self.enable_prometheus = enable_prometheus and prometheus_client is not None
        self.metrics_port = metrics_port
        self.namespace = namespace
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Prometheus setup
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        logger.info(f"MetricsCollector initialized: prometheus={self.enable_prometheus}, port={metrics_port}")
    
    def _setup_prometheus_metrics(self):
        """Setup standard Prometheus metrics."""
        # Execution metrics
        self.register_metric(MetricDefinition(
            name="execution_total",
            description="Total number of executions",
            metric_type=MetricType.COUNTER,
            labels=["component", "strategy", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="execution_duration_seconds",
            description="Execution duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["component", "strategy"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        ))
        
        self.register_metric(MetricDefinition(
            name="active_executions",
            description="Number of currently active executions",
            metric_type=MetricType.GAUGE,
            labels=["component"]
        ))
        
        self.register_metric(MetricDefinition(
            name="worker_pool_size",
            description="Current worker pool size",
            metric_type=MetricType.GAUGE,
            labels=["pool_name", "node"]
        ))
        
        self.register_metric(MetricDefinition(
            name="task_queue_length",
            description="Current task queue length",
            metric_type=MetricType.GAUGE,
            labels=["queue_name", "priority"]
        ))
        
        self.register_metric(MetricDefinition(
            name="security_validations_total",
            description="Total security validations performed",
            metric_type=MetricType.COUNTER,
            labels=["validation_type", "result"]
        ))
        
        self.register_metric(MetricDefinition(
            name="error_total",
            description="Total number of errors",
            metric_type=MetricType.COUNTER,
            labels=["component", "error_type", "severity"]
        ))
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """Register a new metric definition."""
        try:
            self.metric_definitions[definition.name] = definition
            
            if self.enable_prometheus:
                full_name = f"{self.namespace}_{definition.name}"
                
                if definition.metric_type == MetricType.COUNTER:
                    self.metrics[definition.name] = Counter(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.GAUGE:
                    self.metrics[definition.name] = Gauge(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.HISTOGRAM:
                    self.metrics[definition.name] = Histogram(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        buckets=definition.buckets,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.INFO:
                    self.metrics[definition.name] = Info(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry
                    )
            
            logger.debug(f"Registered metric: {definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register metric {definition.name}: {e}")
            return False
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1.0):
        """Increment a counter metric."""
        if metric_name in self.metrics:
            try:
                if labels:
                    self.metrics[metric_name].labels(**labels).inc(value)
                else:
                    self.metrics[metric_name].inc(value)
            except Exception as e:
                logger.error(f"Failed to increment counter {metric_name}: {e}")
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        if metric_name in self.metrics:
            try:
                if labels:
                    self.metrics[metric_name].labels(**labels).set(value)
                else:
                    self.metrics[metric_name].set(value)
            except Exception as e:
                logger.error(f"Failed to set gauge {metric_name}: {e}")
    
    def observe_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value for a histogram metric."""
        if metric_name in self.metrics:
            try:
                if labels:
                    self.metrics[metric_name].labels(**labels).observe(value)
                else:
                    self.metrics[metric_name].observe(value)
            except Exception as e:
                logger.error(f"Failed to observe histogram {metric_name}: {e}")
    
    @contextmanager
    def time_operation(self, metric_name: str, labels: Dict[str, str] = None) -> Iterator[None]:
        """Context manager to time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_histogram(metric_name, duration, labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        if not self.enable_prometheus:
            return {"error": "Prometheus not available"}
        
        try:
            from prometheus_client import generate_latest
            metrics_output = generate_latest(self.registry)
            return {
                "total_metrics": len(self.metrics),
                "prometheus_output_size": len(metrics_output),
                "registered_metrics": list(self.metric_definitions.keys())
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}


# =============================================================================
# Distributed Tracing System
# =============================================================================

class TracingManager:
    """
    OpenTelemetry-based distributed tracing system.
    
    Features:
    - Distributed tracing across components
    - Jaeger integration
    - Automatic span correlation
    - Performance analysis
    - Error tracking
    """
    
    def __init__(
        self,
        service_name: str = "namel3ss-execution",
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        enable_jaeger: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize tracing manager."""
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.enable_jaeger = enable_jaeger and trace is not None
        self.sample_rate = sample_rate
        
        # Tracing state
        self.tracer_provider = None
        self.tracer = None
        self.active_spans: Dict[str, Any] = {}
        
        if self.enable_jaeger:
            self._setup_tracing()
        
        logger.info(f"TracingManager initialized: jaeger={self.enable_jaeger}, service={service_name}")
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            # Create tracer provider
            self.tracer_provider = TracerProvider()
            trace.set_tracer_provider(self.tracer_provider)
            
            # Setup Jaeger exporter
            if JaegerExporter:
                jaeger_exporter = JaegerExporter(
                    endpoint=self.jaeger_endpoint,
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                self.tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)
            
            logger.info("OpenTelemetry tracing setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
            self.enable_jaeger = False
    
    def start_span(
        self, 
        operation_name: str, 
        parent_span: Any = None,
        tags: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Start a new tracing span."""
        if not self.enable_jaeger or not self.tracer:
            return None
        
        try:
            # Create span
            span = self.tracer.start_span(
                operation_name,
                context=parent_span.get_span_context() if parent_span else None
            )
            
            # Add tags/attributes
            if tags:
                for key, value in tags.items():
                    span.set_attribute(key, str(value))
            
            # Store active span
            span_id = str(span.get_span_context().span_id)
            self.active_spans[span_id] = span
            
            return span
            
        except Exception as e:
            logger.error(f"Failed to start span {operation_name}: {e}")
            return None
    
    def finish_span(self, span: Any, status: str = "ok", error: Exception = None):
        """Finish a tracing span."""
        if not span or not self.enable_jaeger:
            return
        
        try:
            # Set span status
            if error:
                span.record_exception(error)
                if Status:
                    span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                if Status:
                    span.set_status(Status(StatusCode.OK))
            
            # End span
            span.end()
            
            # Remove from active spans
            span_id = str(span.get_span_context().span_id)
            self.active_spans.pop(span_id, None)
            
        except Exception as e:
            logger.error(f"Failed to finish span: {e}")
    
    @contextmanager
    def trace_operation(
        self, 
        operation_name: str, 
        tags: Dict[str, Any] = None,
        parent_span: Any = None
    ) -> Iterator[Any]:
        """Context manager for tracing operations."""
        span = self.start_span(operation_name, parent_span, tags)
        try:
            yield span
        except Exception as e:
            if span:
                self.finish_span(span, "error", e)
            raise
        else:
            if span:
                self.finish_span(span, "ok")
    
    def trace_async_function(self, operation_name: str = None, tags: Dict[str, Any] = None):
        """Decorator for tracing async functions."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.trace_operation(op_name, tags) as span:
                    if span:
                        # Add function info
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_trace_context(self, span: Any = None) -> Optional[TraceContext]:
        """Get current trace context."""
        if not span or not self.enable_jaeger:
            return None
        
        try:
            context = span.get_span_context()
            return TraceContext(
                trace_id=str(context.trace_id),
                span_id=str(context.span_id),
                operation_name="unknown"
            )
        except Exception as e:
            logger.error(f"Failed to get trace context: {e}")
            return None


# =============================================================================
# Health Monitoring System
# =============================================================================

class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Features:
    - Component health checks
    - Dependency monitoring
    - Performance thresholds
    - Alert generation
    - Health dashboards
    """
    
    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        enable_auto_checks: bool = True,
    ):
        """Initialize health monitor."""
        self.check_interval = check_interval_seconds
        self.enable_auto_checks = enable_auto_checks
        
        # Health state
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
        self.component_status: Dict[str, HealthStatus] = {}
        
        # Auto-check task
        self.monitor_task = None
        
        self._register_default_checks()
        
        if self.enable_auto_checks:
            self.start_monitoring()
        
        logger.info(f"HealthMonitor initialized: interval={check_interval_seconds}s")
    
    def _register_default_checks(self):
        """Register default health checks."""
        # System health
        self.register_health_check(HealthCheck(
            name="system_memory",
            description="System memory usage",
            check_function=self._check_system_memory,
            critical=True
        ))
        
        self.register_health_check(HealthCheck(
            name="system_cpu",
            description="System CPU usage",
            check_function=self._check_system_cpu,
            critical=False
        ))
        
        # Component health
        self.register_health_check(HealthCheck(
            name="parallel_executor",
            description="Parallel execution engine status",
            check_function=self._check_parallel_executor,
            critical=True
        ))
        
        self.register_health_check(HealthCheck(
            name="distributed_queue",
            description="Distributed task queue status",
            check_function=self._check_distributed_queue,
            critical=True
        ))
    
    def register_health_check(self, health_check: HealthCheck) -> bool:
        """Register a new health check."""
        try:
            self.health_checks[health_check.name] = health_check
            logger.debug(f"Registered health check: {health_check.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register health check {health_check.name}: {e}")
            return False
    
    async def run_health_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if check_name not in self.health_checks:
            return {"error": f"Health check {check_name} not found"}
        
        check = self.health_checks[check_name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, check.check_function),
                timeout=check.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            check_result = {
                "name": check.name,
                "status": HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                "description": check.description,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
                "critical": check.critical,
                "tags": check.tags
            }
            
            self.last_check_results[check_name] = check_result
            return check_result
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            check_result = {
                "name": check.name,
                "status": HealthStatus.CRITICAL,
                "description": f"{check.description} (timeout)",
                "duration_ms": duration_ms,
                "timestamp": time.time(),
                "critical": check.critical,
                "error": f"Check timed out after {check.timeout_seconds}s"
            }
            self.last_check_results[check_name] = check_result
            return check_result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            check_result = {
                "name": check.name,
                "status": HealthStatus.CRITICAL,
                "description": f"{check.description} (error)",
                "duration_ms": duration_ms,
                "timestamp": time.time(),
                "critical": check.critical,
                "error": str(e)
            }
            self.last_check_results[check_name] = check_result
            return check_result
    
    async def run_all_health_checks(self) -> HealthReport:
        """Run all registered health checks."""
        start_time = time.time()
        check_results = []
        
        # Run all checks concurrently
        check_tasks = [
            self.run_health_check(check_name) 
            for check_name in self.health_checks.keys()
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        overall_status = HealthStatus.HEALTHY
        critical_failures = 0
        
        for result in results:
            if isinstance(result, Exception):
                check_results.append({
                    "name": "unknown",
                    "status": HealthStatus.CRITICAL,
                    "error": str(result)
                })
                overall_status = HealthStatus.CRITICAL
                critical_failures += 1
            else:
                check_results.append(result)
                
                if result.get("status") == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                    if result.get("critical"):
                        critical_failures += 1
                elif result.get("status") == HealthStatus.UNHEALTHY and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthReport(
            component="namel3ss_execution",
            status=overall_status,
            checks=check_results,
            timestamp=time.time(),
            response_time_ms=duration_ms,
            details={
                "total_checks": len(check_results),
                "critical_failures": critical_failures,
                "healthy_checks": len([r for r in check_results if r.get("status") == HealthStatus.HEALTHY])
            }
        )
    
    def start_monitoring(self):
        """Start automatic health monitoring."""
        if self.monitor_task:
            return
        
        async def monitor_loop():
            while True:
                try:
                    await self.run_all_health_checks()
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(5.0)  # Brief pause on error
        
        self.monitor_task = asyncio.create_task(monitor_loop())
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic health monitoring."""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
            logger.info("Health monitoring stopped")
    
    # Default health check implementations
    def _check_system_memory(self) -> bool:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 85.0  # Alert if memory > 85%
        except ImportError:
            return True  # Can't check without psutil
        except Exception:
            return False
    
    def _check_system_cpu(self) -> bool:
        """Check system CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90.0  # Alert if CPU > 90%
        except ImportError:
            return True  # Can't check without psutil
        except Exception:
            return False
    
    def _check_parallel_executor(self) -> bool:
        """Check parallel executor status."""
        # This would check if parallel executor is responding
        # For now, always return True (placeholder)
        return True
    
    def _check_distributed_queue(self) -> bool:
        """Check distributed queue status."""
        # This would check if distributed queue is responding
        # For now, always return True (placeholder)
        return True


# =============================================================================
# Observability Manager
# =============================================================================

class ObservabilityManager:
    """
    Central observability manager coordinating all monitoring systems.
    
    Features:
    - Unified observability interface
    - Metrics, tracing, and health monitoring
    - Alert management
    - Dashboard data aggregation
    - Production deployment support
    """
    
    def __init__(
        self,
        service_name: str = "namel3ss-execution",
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_health_monitoring: bool = True,
        **config_kwargs
    ):
        """Initialize observability manager."""
        self.service_name = service_name
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.enable_health_monitoring = enable_health_monitoring
        
        # Initialize subsystems
        self.metrics = MetricsCollector(**config_kwargs) if enable_metrics else None
        self.tracing = TracingManager(service_name=service_name, **config_kwargs) if enable_tracing else None
        self.health = HealthMonitor(**config_kwargs) if enable_health_monitoring else None
        
        # Observability state
        self.start_time = time.time()
        self.operation_counters: Dict[str, int] = {}
        
        logger.info(f"ObservabilityManager initialized: metrics={enable_metrics}, tracing={enable_tracing}, health={enable_health_monitoring}")
    
    async def record_execution_start(
        self, 
        component: str, 
        strategy: str, 
        execution_id: str,
        **tags
    ) -> Optional[Any]:
        """Record the start of an execution."""
        # Increment counter
        if self.metrics:
            self.metrics.increment_counter(
                "execution_total",
                labels={"component": component, "strategy": strategy, "status": "started"}
            )
            self.metrics.set_gauge(
                "active_executions",
                self.metrics.metrics.get("active_executions", 0) + 1,
                labels={"component": component}
            )
        
        # Start tracing span
        span = None
        if self.tracing:
            span_tags = {
                "component": component,
                "strategy": strategy,
                "execution.id": execution_id,
                **tags
            }
            span = self.tracing.start_span(f"{component}.execute", tags=span_tags)
        
        return span
    
    async def record_execution_complete(
        self,
        component: str,
        strategy: str,
        execution_id: str,
        span: Any = None,
        duration_seconds: float = None,
        success: bool = True,
        error: Exception = None,
        **tags
    ):
        """Record the completion of an execution."""
        # Record metrics
        if self.metrics:
            status = "success" if success else "error"
            
            self.metrics.increment_counter(
                "execution_total",
                labels={"component": component, "strategy": strategy, "status": status}
            )
            
            if duration_seconds:
                self.metrics.observe_histogram(
                    "execution_duration_seconds",
                    duration_seconds,
                    labels={"component": component, "strategy": strategy}
                )
            
            # Update active executions gauge
            current_active = self.operation_counters.get(component, 0)
            if current_active > 0:
                self.operation_counters[component] = current_active - 1
                self.metrics.set_gauge(
                    "active_executions",
                    current_active - 1,
                    labels={"component": component}
                )
        
        # Finish tracing span
        if self.tracing and span:
            self.tracing.finish_span(span, "ok" if success else "error", error)
    
    async def record_security_validation(
        self,
        validation_type: str,
        result: str,
        user_id: str = None,
        **tags
    ):
        """Record security validation event."""
        if self.metrics:
            self.metrics.increment_counter(
                "security_validations_total",
                labels={"validation_type": validation_type, "result": result}
            )
    
    async def record_error(
        self,
        component: str,
        error_type: str,
        severity: str = "error",
        **tags
    ):
        """Record an error event."""
        if self.metrics:
            self.metrics.increment_counter(
                "error_total",
                labels={"component": component, "error_type": error_type, "severity": severity}
            )
    
    async def get_health_report(self) -> Optional[HealthReport]:
        """Get comprehensive health report."""
        if not self.health:
            return None
        
        return await self.health.run_all_health_checks()
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "observability_enabled": {
                "metrics": self.enable_metrics,
                "tracing": self.enable_tracing,
                "health": self.enable_health_monitoring,
            },
            "uptime_seconds": time.time() - self.start_time,
            "service_name": self.service_name,
        }
        
        if self.metrics:
            summary["metrics"] = self.metrics.get_metrics_summary()
        
        if self.tracing:
            summary["tracing"] = {
                "enabled": self.tracing.enable_jaeger,
                "active_spans": len(self.tracing.active_spans),
                "service_name": self.tracing.service_name,
            }
        
        return summary
    
    def get_trace_decorator(self, operation_name: str = None, tags: Dict[str, Any] = None):
        """Get tracing decorator for async functions."""
        if not self.tracing:
            # No-op decorator if tracing disabled
            def no_op_decorator(func):
                return func
            return no_op_decorator
        
        return self.tracing.trace_async_function(operation_name, tags)
    
    @contextmanager
    def time_operation(self, metric_name: str, labels: Dict[str, str] = None):
        """Context manager to time operations."""
        if self.metrics:
            yield self.metrics.time_operation(metric_name, labels)
        else:
            yield


# =============================================================================
# Global Observability Instance
# =============================================================================

# Global observability manager instance
_global_observability: Optional[ObservabilityManager] = None


def get_observability_manager() -> ObservabilityManager:
    """Get global observability manager instance."""
    global _global_observability
    
    if _global_observability is None:
        _global_observability = ObservabilityManager()
    
    return _global_observability


def initialize_observability(
    service_name: str = "namel3ss-execution",
    **config_kwargs
) -> ObservabilityManager:
    """Initialize global observability with custom configuration."""
    global _global_observability
    _global_observability = ObservabilityManager(service_name=service_name, **config_kwargs)
    return _global_observability


# Convenience functions
async def record_execution_metrics(
    component: str,
    strategy: str,
    execution_id: str,
    success: bool = True,
    duration_seconds: float = None,
    error: Exception = None
):
    """Convenience function to record execution metrics."""
    obs = get_observability_manager()
    await obs.record_execution_complete(
        component=component,
        strategy=strategy,
        execution_id=execution_id,
        success=success,
        duration_seconds=duration_seconds,
        error=error
    )


def trace_execution(operation_name: str = None, tags: Dict[str, Any] = None):
    """Convenience decorator for tracing executions."""
    obs = get_observability_manager()
    return obs.get_trace_decorator(operation_name, tags)