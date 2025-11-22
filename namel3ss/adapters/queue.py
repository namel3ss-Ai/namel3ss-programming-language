"""Queue adapter for async task processing via Celery/RQ.

Provides typed async task queuing with result tracking and retry support.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import (
    AdapterConfig,
    AdapterType,
    BaseAdapter,
    AdapterExecutionError,
    AdapterValidationError,
)


class QueueBackend(str, Enum):
    """Supported queue backends."""
    CELERY = "celery"
    RQ = "rq"
    KAFKA = "kafka"


class QueueAdapterConfig(AdapterConfig):
    """Configuration for queue adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.QUEUE)
    
    # Backend settings
    backend: QueueBackend = Field(..., description="Queue backend type")
    broker_url: str = Field(..., description="Message broker URL")
    result_backend: Optional[str] = Field(None, description="Result backend URL")
    
    # Queue settings
    queue_name: str = Field(default="default", description="Queue name")
    routing_key: Optional[str] = Field(None, description="Routing key (Celery)")
    
    # Task settings
    task_name: str = Field(..., description="Task identifier")
    task_timeout: float = Field(default=300.0, ge=1.0, description="Task execution timeout")
    
    # Retry settings
    task_max_retries: int = Field(default=3, ge=0, description="Max task retries")
    task_retry_delay: float = Field(default=60.0, ge=0.0, description="Retry delay in seconds")
    
    # Result settings
    result_expires: int = Field(default=3600, ge=0, description="Result expiration (seconds)")
    track_started: bool = Field(default=True, description="Track task started state")


class CeleryQueueAdapter(BaseAdapter):
    """Celery-based queue adapter.
    
    Integrates with Celery for distributed task processing.
    
    Example:
        Define Celery task:
        >>> # tasks.py
        >>> from celery import Celery
        >>> 
        >>> app = Celery('tasks', broker='redis://localhost:6379/0')
        >>> 
        >>> @app.task(name='process_document')
        >>> def process_document(doc_id: int, text: str) -> dict:
        ...     # Process document
        ...     return {"doc_id": doc_id, "status": "processed"}
        
        Use from N3:
        ```n3
        tool "queue_document" {
          adapter: "queue"
          backend: "celery"
          broker_url: "redis://localhost:6379/0"
          queue_name: "documents"
          task_name: "process_document"
        }
        
        chain "enqueue_processing" {
          call: "queue_document"
          inputs: {
            doc_id: {{document.id}}
            text: {{document.text}}
          }
        }
        ```
    """
    
    def __init__(self, config: QueueAdapterConfig):
        try:
            from celery import Celery
            self._celery_module = Celery
            self._has_celery = True
        except ImportError:
            self._has_celery = False
        
        super().__init__(config)
        self.config: QueueAdapterConfig = config
        self._app = None
        
        if config.backend == QueueBackend.CELERY:
            if not self._has_celery:
                raise AdapterExecutionError(
                    "Celery not installed. Install with: pip install celery",
                    adapter_name=config.name,
                    adapter_type=config.adapter_type,
                )
            self._setup_celery()
    
    def _setup_celery(self):
        """Setup Celery app."""
        self._app = self._celery_module(
            'n3_queue_adapter',
            broker=self.config.broker_url,
            backend=self.config.result_backend,
        )
        
        # Configure app
        self._app.conf.update(
            task_track_started=self.config.track_started,
            result_expires=self.config.result_expires,
            task_time_limit=self.config.task_timeout,
            task_soft_time_limit=self.config.task_timeout * 0.9,
        )
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Enqueue task for async processing."""
        if not self._app:
            raise AdapterExecutionError(
                "Queue backend not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        try:
            # Send task to queue
            result = self._app.send_task(
                self.config.task_name,
                kwargs=inputs,
                queue=self.config.queue_name,
                routing_key=self.config.routing_key,
                retry=self.config.task_max_retries > 0,
                retry_policy={
                    'max_retries': self.config.task_max_retries,
                    'interval_start': self.config.task_retry_delay,
                    'interval_step': self.config.task_retry_delay,
                    'interval_max': self.config.task_retry_delay * 4,
                }
            )
            
            return {
                "task_id": result.id,
                "status": "queued",
                "queue": self.config.queue_name,
                "task": self.config.task_name,
            }
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Failed to enqueue task: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status by ID.
        
        Args:
            task_id: Task ID returned from execute()
        
        Returns:
            Task status dict with state and result
        
        Example:
            >>> result = adapter.execute(doc_id=123)
            >>> status = adapter.get_task_status(result['task_id'])
            >>> print(status['state'])  # PENDING, STARTED, SUCCESS, FAILURE
        """
        if not self._app:
            raise AdapterExecutionError(
                "Queue backend not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        from celery.result import AsyncResult
        
        task = AsyncResult(task_id, app=self._app)
        
        return {
            "task_id": task_id,
            "state": task.state,
            "ready": task.ready(),
            "successful": task.successful(),
            "failed": task.failed(),
            "result": task.result if task.ready() else None,
        }
    
    def wait_for_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task to complete and return result.
        
        Args:
            task_id: Task ID
            timeout: Max wait time (uses adapter timeout if not specified)
        
        Returns:
            Task result
        
        Raises:
            AdapterTimeoutError: Task didn't complete in time
            AdapterExecutionError: Task failed
        
        Example:
            >>> result = adapter.execute(doc_id=123)
            >>> output = adapter.wait_for_result(result['task_id'], timeout=60.0)
        """
        if not self._app:
            raise AdapterExecutionError(
                "Queue backend not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        from celery.result import AsyncResult
        from celery.exceptions import TimeoutError as CeleryTimeoutError
        
        task = AsyncResult(task_id, app=self._app)
        timeout = timeout or self.config.timeout
        
        try:
            result = task.get(timeout=timeout)
            return result
        
        except CeleryTimeoutError:
            from .base import AdapterTimeoutError
            raise AdapterTimeoutError(
                f"Task {task_id} did not complete within {timeout}s",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Task {task_id} failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )


class RQQueueAdapter(BaseAdapter):
    """RQ (Redis Queue) based adapter.
    
    Simpler alternative to Celery for Redis-based queuing.
    
    Example:
        Define RQ job:
        >>> # jobs.py
        >>> def process_data(data: dict) -> dict:
        ...     # Process data
        ...     return {"status": "processed", "data": data}
        
        Use from N3:
        ```n3
        tool "queue_job" {
          adapter: "queue"
          backend: "rq"
          broker_url: "redis://localhost:6379/0"
          queue_name: "default"
          task_name: "jobs.process_data"
        }
        ```
    """
    
    def __init__(self, config: QueueAdapterConfig):
        from namel3ss.features import has_redis
        
        try:
            import rq
            from redis import Redis
            self._rq = rq
            self._redis = Redis
            self._has_rq = True
        except ImportError:
            self._has_rq = False
        
        super().__init__(config)
        self.config: QueueAdapterConfig = config
        self._queue = None
        
        if config.backend == QueueBackend.RQ:
            if not self._has_rq or not has_redis():
                raise AdapterExecutionError(
                    "RQ support requires the 'redis' extra.\n"
                    "Install with: pip install 'namel3ss[redis]'",
                    adapter_name=config.name,
                    adapter_type=config.adapter_type,
                )
            self._setup_rq()
    
    def _setup_rq(self):
        """Setup RQ queue."""
        # Parse Redis URL
        redis_conn = self._redis.from_url(self.config.broker_url)
        self._queue = self._rq.Queue(
            self.config.queue_name,
            connection=redis_conn,
        )
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Enqueue job to RQ."""
        if not self._queue:
            raise AdapterExecutionError(
                "RQ queue not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        try:
            # Import and enqueue function
            job = self._queue.enqueue(
                self.config.task_name,
                kwargs=inputs,
                job_timeout=self.config.task_timeout,
                result_ttl=self.config.result_expires,
                failure_ttl=86400,  # 24 hours
            )
            
            return {
                "job_id": job.id,
                "status": "queued",
                "queue": self.config.queue_name,
                "task": self.config.task_name,
            }
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Failed to enqueue job: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )


def create_queue_adapter(config: QueueAdapterConfig) -> BaseAdapter:
    """Factory function to create appropriate queue adapter.
    
    Args:
        config: Queue adapter configuration
    
    Returns:
        CeleryQueueAdapter or RQQueueAdapter based on backend
    
    Example:
        >>> config = QueueAdapterConfig(
        ...     name="my_queue",
        ...     backend="celery",
        ...     broker_url="redis://localhost:6379/0",
        ...     task_name="tasks.process"
        ... )
        >>> adapter = create_queue_adapter(config)
    """
    if config.backend == QueueBackend.CELERY:
        return CeleryQueueAdapter(config)
    elif config.backend == QueueBackend.RQ:
        return RQQueueAdapter(config)
    else:
        raise AdapterValidationError(
            f"Unsupported queue backend: {config.backend}",
            adapter_name=config.name,
            adapter_type=config.adapter_type,
        )
