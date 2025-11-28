"""World-class Ollama provider for local model deployment and management.

This module provides a production-grade Ollama integration with:
- Configurable server URLs via environment variables
- Intelligent caching for model availability and health checks
- Comprehensive metrics and observability
- Developer-friendly error messages with actionable guidance
- Editor integration hooks for VSCode and other tools
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union
from urllib.parse import urljoin

import httpx

from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric
from ..base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from ..factory import register_provider_class

logger = get_logger(__name__)

# Environment variable for base URL override
OLLAMA_BASE_URL_ENV = "NAMEL3SS_OLLAMA_BASE_URL"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Cache configuration
MODEL_CACHE_TTL_SECONDS = 60  # Cache model availability for 1 minute
HEALTH_CHECK_MIN_INTERVAL = 30  # Minimum seconds between health checks


@dataclass
class CacheEntry:
    """Cache entry with value and timestamp."""
    value: Any
    timestamp: float
    
    def is_stale(self, ttl: float) -> bool:
        """Check if entry is older than TTL."""
        return (time.time() - self.timestamp) > ttl


class ModelAvailabilityCache:
    """Thread-safe cache for model availability checks.
    
    Reduces unnecessary API calls to Ollama server by caching
    model availability results with a configurable TTL.
    """
    
    def __init__(self, ttl: float = MODEL_CACHE_TTL_SECONDS):
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, cache_key: str) -> Optional[bool]:
        """Get cached availability if not stale."""
        async with self._lock:
            entry = self._cache.get(cache_key)
            if entry and not entry.is_stale(self.ttl):
                return entry.value
            return None
    
    async def set(self, cache_key: str, available: bool) -> None:
        """Cache model availability."""
        async with self._lock:
            self._cache[cache_key] = CacheEntry(
                value=available,
                timestamp=time.time()
            )
    
    async def invalidate(self, cache_key: str) -> None:
        """Remove a cache entry."""
        async with self._lock:
            self._cache.pop(cache_key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()


class OllamaError(ProviderError):
    """Specialized error for Ollama-specific failures with helpful guidance."""
    
    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Create Ollama error with context and guidance.
        
        Args:
            message: Primary error message
            model: Model name if relevant
            base_url: Ollama server URL if relevant
            suggestion: Actionable suggestion for fixing the issue
            original_error: Underlying exception if any
        """
        full_message = message
        
        if model:
            full_message += f"\nModel: {model}"
        if base_url:
            full_message += f"\nServer: {base_url}"
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"
        
        super().__init__(full_message)
        self.model = model
        self.base_url = base_url
        self.suggestion = suggestion
        self.original_error = original_error


class OllamaModelManager:
    """Manager for Ollama model pulling and lifecycle with intelligent caching."""
    
    def __init__(self, base_url: str, cache: Optional[ModelAvailabilityCache] = None):
        self.base_url = base_url
        self._pulled_models: Set[str] = set()
        self._cache = cache or ModelAvailabilityCache()
        
    def _cache_key(self, model_name: str) -> str:
        """Generate cache key for model."""
        return f"{self.base_url}:{model_name}"
        
    async def ensure_model_available(
        self,
        model_name: str,
        *,
        auto_pull: bool = True,
        pull_timeout: int = 600
    ) -> None:
        """Ensure model is pulled and available.
        
        Args:
            model_name: Name of the model (e.g., "llama3:8b")
            auto_pull: Whether to automatically pull missing models
            pull_timeout: Timeout for model pulling in seconds
            
        Raises:
            OllamaError: If model is unavailable and cannot be pulled
        """
        # Check cache first
        cache_key = self._cache_key(model_name)
        cached_availability = await self._cache.get(cache_key)
        
        if cached_availability is True:
            logger.debug(f"Model {model_name} available (cached)")
            return
        
        # Check if model is available on server
        start_time = time.time()
        is_available = await self._is_model_available(model_name)
        check_duration = time.time() - start_time
        
        record_metric(
            "ollama.model.check_duration",
            check_duration,
            {"model": model_name}
        )
        
        if is_available:
            await self._cache.set(cache_key, True)
            self._pulled_models.add(model_name)
            logger.info(f"Model {model_name} is available")
            return
        
        # Model not available
        if not auto_pull:
            raise OllamaError(
                f"Ollama model '{model_name}' is not available locally.",
                model=model_name,
                base_url=self.base_url,
                suggestion=f"Run: ollama pull {model_name}\n"
                           f"Or enable auto_pull in your configuration."
            )
        
        # Try to pull the model
        logger.info(f"Model {model_name} not found, pulling...")
        try:
            await self._pull_model(model_name, timeout=pull_timeout)
            await self._cache.set(cache_key, True)
            self._pulled_models.add(model_name)
        except Exception as e:
            raise OllamaError(
                f"Failed to auto-pull Ollama model '{model_name}'.",
                model=model_name,
                base_url=self.base_url,
                suggestion=f"Try manually: ollama pull {model_name}\n"
                           f"Check disk space and network connectivity.",
                original_error=e
            )
    
    async def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = data.get('models', [])
                
                for model in models:
                    if model.get('name') == model_name:
                        return True
                return False
                
        except httpx.ConnectError as e:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to check Ollama model availability: {e}")
            return False
    
    async def _pull_model(self, model_name: str, timeout: int = 600) -> None:
        """Pull model from Ollama registry with progress tracking.
        
        Args:
            model_name: Name of the model to pull
            timeout: Maximum time to wait for pull in seconds
            
        Raises:
            OllamaError: If pull fails
        """
        pull_start = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    response.raise_for_status()
                    
                    last_log_time = time.time()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                status_data = json.loads(line)
                                status = status_data.get('status', '')
                                
                                # Log progress periodically (every 5 seconds)
                                current_time = time.time()
                                if current_time - last_log_time >= 5.0:
                                    logger.info(f"Ollama pulling {model_name}: {status}")
                                    last_log_time = current_time
                                
                                if status_data.get('status') == 'success':
                                    pull_duration = time.time() - pull_start
                                    logger.info(
                                        f"Successfully pulled Ollama model {model_name} "
                                        f"in {pull_duration:.1f}s"
                                    )
                                    record_metric(
                                        "ollama.model.pull_duration",
                                        pull_duration,
                                        {"model": model_name}
                                    )
                                    return
                            except json.JSONDecodeError:
                                continue
                
        except httpx.TimeoutException:
            raise OllamaError(
                f"Timeout while pulling model '{model_name}' (>{timeout}s).",
                model=model_name,
                base_url=self.base_url,
                suggestion="Model pull timed out. Large models can take several minutes.\n"
                           "Try increasing pull_timeout or pull manually."
            )
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models.
        
        Returns:
            List of model metadata dictionaries
            
        Raises:
            OllamaError: If server is unreachable
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = data.get('models', [])
                
                record_metric(
                    "ollama.models.count",
                    len(models),
                    {"base_url": self.base_url}
                )
                
                return models
                
        except httpx.ConnectError:
            raise OllamaError(
                f"Could not reach Ollama server at {self.base_url}",
                base_url=self.base_url,
                suggestion="Ensure Ollama is installed and running.\n"
                           "Install from: https://ollama.ai\n"
                           "Start with: ollama serve"
            )
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise OllamaError(
                "Failed to list Ollama models.",
                base_url=self.base_url,
                original_error=e
            )
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise OllamaError(
                "Failed to list Ollama models.",
                base_url=self.base_url,
                original_error=e
            )
    
    async def delete_model(self, model_name: str) -> None:
        """Delete a model from Ollama.
        
        Args:
            model_name: Name of the model to delete
            
        Raises:
            OllamaError: If deletion fails
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/delete",
                    json={"name": model_name}
                )
                response.raise_for_status()
                
                # Invalidate cache
                cache_key = self._cache_key(model_name)
                await self._cache.invalidate(cache_key)
                self._pulled_models.discard(model_name)
                
                logger.info(f"Deleted Ollama model: {model_name}")
                record_metric(
                    "ollama.model.deleted",
                    1,
                    {"model": model_name}
                )
                
        except Exception as e:
            logger.error(f"Failed to delete Ollama model {model_name}: {e}")
            raise OllamaError(
                f"Failed to delete model '{model_name}'.",
                model=model_name,
                base_url=self.base_url,
                original_error=e
            )


class OllamaServerManager:
    """Manager for Ollama server deployment and lifecycle with health monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 11434)
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Health check throttling
        self._last_health_check: float = 0
        self._health_check_interval: float = config.get('health_check_interval', HEALTH_CHECK_MIN_INTERVAL)
        self._cached_health: Optional[Dict[str, Any]] = None
        
    async def start_server(self) -> None:
        """Start Ollama server if not running.
        
        Raises:
            OllamaError: If server fails to start
        """
        if await self._is_server_running():
            logger.info("Ollama server already running")
            return
            
        logger.info("Starting Ollama server")
        
        try:
            # Set environment variables for Ollama
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f"{self.host}:{self.port}"
            
            if 'gpu_layers' in self.config:
                env['OLLAMA_NUM_GPU'] = str(self.config['gpu_layers'])
            
            if 'num_thread' in self.config:
                env['OLLAMA_NUM_THREAD'] = str(self.config['num_thread'])
                
            # Start Ollama serve
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Wait for server to be ready
            await self._wait_for_server_ready()
            logger.info(f"Ollama server started at {self.base_url}")
            record_metric("ollama.server.started", 1, {"base_url": self.base_url})
            
        except FileNotFoundError:
            raise OllamaError(
                "Ollama executable not found.",
                base_url=self.base_url,
                suggestion="Install Ollama from: https://ollama.ai\n"
                           "Ensure 'ollama' is in your PATH."
            )
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            raise OllamaError(
                "Failed to start Ollama server.",
                base_url=self.base_url,
                suggestion="Check if Ollama is properly installed and the port is available.",
                original_error=e
            )
    
    async def _is_server_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def _wait_for_server_ready(self, max_wait: int = 60) -> None:
        """Wait for Ollama server to be ready.
        
        Args:
            max_wait: Maximum seconds to wait
            
        Raises:
            OllamaError: If server doesn't start in time
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self._is_server_running():
                return
            await asyncio.sleep(2)
        
        raise OllamaError(
            f"Ollama server did not start within {max_wait} seconds.",
            base_url=self.base_url,
            suggestion="Check Ollama logs for startup errors.\n"
                       "Try starting manually: ollama serve"
        )
    
    async def stop_server(self) -> None:
        """Stop Ollama server gracefully."""
        if self.process:
            logger.info("Stopping Ollama server")
            self.process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_end()),
                    timeout=30.0
                )
                record_metric("ollama.server.stopped", 1, {"base_url": self.base_url})
            except asyncio.TimeoutError:
                logger.warning("Ollama server did not stop gracefully, forcing kill")
                self.process.kill()
            
            self.process = None
    
    async def _wait_for_process_end(self) -> None:
        """Wait for process to end."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def health_check(self, *, force: bool = False) -> Dict[str, Any]:
        """Check Ollama server health with intelligent throttling.
        
        Args:
            force: Force health check even if recently checked
            
        Returns:
            Health status dictionary
            
        Raises:
            OllamaError: If server is unreachable
        """
        current_time = time.time()
        
        # Return cached health if recent (unless forced)
        if not force and self._cached_health is not None:
            if (current_time - self._last_health_check) < self._health_check_interval:
                logger.debug(f"Using cached health check (age: {current_time - self._last_health_check:.1f}s)")
                return self._cached_health
        
        # Perform actual health check
        try:
            check_start = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                check_duration = time.time() - check_start
                
                health_info = {
                    'status': 'healthy',
                    'models': data.get('models', []),
                    'base_url': self.base_url,
                    'response_time_ms': check_duration * 1000,
                    'checked_at': current_time,
                }
                
                # Update cache
                self._cached_health = health_info
                self._last_health_check = current_time
                
                record_metric(
                    "ollama.health.check_duration",
                    check_duration,
                    {"base_url": self.base_url, "status": "healthy"}
                )
                
                return health_info
                
        except httpx.ConnectError:
            error_info = {
                'status': 'unreachable',
                'base_url': self.base_url,
                'checked_at': current_time,
            }
            self._cached_health = error_info
            self._last_health_check = current_time
            
            raise OllamaError(
                f"Could not reach Ollama server at {self.base_url}",
                base_url=self.base_url,
                suggestion="Ensure Ollama is running.\n"
                           "Install: https://ollama.ai\n"
                           "Start: ollama serve"
            )
        except Exception as e:
            raise OllamaError(
                "Ollama health check failed.",
                base_url=self.base_url,
                original_error=e
            )


class OllamaProvider(N3Provider):
    """
    World-class Ollama provider for local model deployment.
    
    Features:
    - Configurable server URL via environment variables
    - Intelligent caching for model availability and health checks
    - Comprehensive metrics and observability
    - Developer-friendly error messages with actionable guidance
    - Streaming response support with token tracking
    - GPU acceleration and optimization
    - Integration with N3 observability system
    - Support for all Ollama-compatible models
    
    Configuration:
        - model: Ollama model name (e.g., "llama3:8b", "mistral", "codellama")
        - base_url: Ollama server URL (default: from NAMEL3SS_OLLAMA_BASE_URL env or http://localhost:11434)
        - host: Server host (default: 127.0.0.1) - used if base_url not provided
        - port: Server port (default: 11434) - used if base_url not provided
        - auto_pull_model: Whether to auto-pull missing models (default: True)
        - auto_start_server: Whether to auto-start Ollama server (default: False)
        - model_cache_ttl: Cache TTL for model availability in seconds (default: 60)
        - health_check_interval: Minimum seconds between health checks (default: 30)
        - gpu_layers: Number of layers to run on GPU
        - num_thread: Number of CPU threads to use
        - keep_alive: How long to keep model loaded (default: "5m")
        
    Deployment Configuration:
        - num_gpu: Number of GPUs to use
        - num_thread: Number of threads
        - num_ctx: Context window size
        - repeat_penalty: Repetition penalty
        - temperature: Sampling temperature
        - top_k: Top-k sampling
        - top_p: Top-p (nucleus) sampling
        
    Environment Variables:
        - NAMEL3SS_OLLAMA_BASE_URL: Override default Ollama server URL
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Ollama-specific configuration
        self.deployment_config = self.config.get('deployment_config', {})
        self.auto_pull_model = self.config.get('auto_pull_model', True)
        self.auto_start_server = self.config.get('auto_start_server', False)  # Default to False for safety
        
        # Base URL configuration with environment variable support
        self.base_url = self._determine_base_url()
        
        # Base URL configuration with environment variable support
        self.base_url = self._determine_base_url()
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.top_k = self.config.get('top_k', 40)
        self.top_p = float(self.config.get('top_p', 0.9))
        self.repeat_penalty = float(self.config.get('repeat_penalty', 1.1))
        self.num_ctx = self.config.get('num_ctx', 2048)
        self.keep_alive = self.config.get('keep_alive', "5m")
        
        # Cache configuration
        model_cache_ttl = self.config.get('model_cache_ttl', MODEL_CACHE_TTL_SECONDS)
        self._cache = ModelAvailabilityCache(ttl=model_cache_ttl)
        
        # HTTP client - reused for performance
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Managers
        server_config = {
            **self.deployment_config,
            'host': self.base_url.split('://')[-1].split(':')[0],  # Extract host from URL
            'port': int(self.base_url.split(':')[-1].rstrip('/')),  # Extract port from URL
            'health_check_interval': self.config.get('health_check_interval', HEALTH_CHECK_MIN_INTERVAL),
        }
        self._server_manager = OllamaServerManager(server_config)
        self._model_manager = OllamaModelManager(self.base_url, cache=self._cache)
        
        # Health tracking
        self._last_health_check = 0
        self._health_check_interval = self.config.get('health_check_interval', HEALTH_CHECK_MIN_INTERVAL)
        
        logger.info(f"Initialized Ollama provider '{name}' with model '{model}' at {self.base_url}")
    
    def _determine_base_url(self) -> str:
        """Determine base URL from config or environment.
        
        Priority:
        1. Explicit base_url in config
        2. NAMEL3SS_OLLAMA_BASE_URL environment variable
        3. Constructed from host:port in config
        4. Default URL
        
        Returns:
            Resolved base URL
        """
        # Explicit base_url in config
        if 'base_url' in self.config:
            url = self.config['base_url']
            logger.debug(f"Using configured base_url: {url}")
            return url
        
        # Environment variable
        env_url = os.getenv(OLLAMA_BASE_URL_ENV)
        if env_url:
            logger.debug(f"Using {OLLAMA_BASE_URL_ENV}: {env_url}")
            return env_url
        
        # Construct from host and port
        if 'host' in self.config or 'port' in self.config:
            host = self.config.get('host', '127.0.0.1')
            port = self.config.get('port', 11434)
            url = f"http://{host}:{port}"
            logger.debug(f"Constructed URL from host:port: {url}")
            return url
        
        # Default
        logger.debug(f"Using default URL: {DEFAULT_OLLAMA_URL}")
        return DEFAULT_OLLAMA_URL
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client for connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, read=300.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
            )
            logger.debug("Created new HTTP client for Ollama")
        return self._http_client
    
    async def _ensure_ready(self) -> None:
        """Ensure Ollama server and model are ready with throttled checks."""
        current_time = time.time()
        
        # Skip if recently checked (throttling)
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        # Ensure server is running (if auto-start enabled)
        if self.auto_start_server:
            try:
                await self._server_manager.start_server()
            except OllamaError as e:
                # Re-raise with context
                raise OllamaError(
                    f"Cannot start Ollama server for model '{self.model}'.",
                    model=self.model,
                    base_url=self.base_url,
                    suggestion=e.suggestion or "Check server configuration and logs.",
                    original_error=e.original_error
                )
        
        # Ensure model is available
        try:
            await self._model_manager.ensure_model_available(
                self.model,
                auto_pull=self.auto_pull_model
            )
        except OllamaError:
            # Already has good error message, just re-raise
            raise
        
        self._last_health_check = current_time
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> ProviderResponse:
        """
        Generate text completion using Ollama.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            ProviderResponse with generated text
            
        Raises:
            OllamaError: If generation fails with helpful context
        """
        await self._ensure_ready()
        
        # Build request payload
        payload = self._build_request_payload(messages, stream=False, **kwargs)
        
        start_time = time.time()
        
        try:
            client = await self._get_http_client()
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            duration = time.time() - start_time
            
            # Extract token counts if available
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate throughput
            tokens_per_second = completion_tokens / duration if duration > 0 else 0
            
            # Record comprehensive metrics
            record_metric("ollama.request.duration", duration, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.tokens.total", total_tokens, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.tokens.prompt", prompt_tokens, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.tokens.completion", completion_tokens, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.throughput.tokens_per_second", tokens_per_second, {
                "provider": self.name,
                "model": self.model
            })
            
            logger.info(
                f"Ollama generate completed: {total_tokens} tokens in {duration:.2f}s "
                f"({tokens_per_second:.1f} tok/s)"
            )
            
            return ProviderResponse(
                model=result.get('model', self.model),
                output_text=result['message']['content'],
                raw=result,
                metadata={
                    'created_at': result.get('created_at'),
                    'done': result.get('done', True),
                    'provider': 'ollama',
                    'duration': duration,
                    'tokens': {
                        'prompt': prompt_tokens,
                        'completion': completion_tokens,
                        'total': total_tokens,
                    },
                    'throughput': {
                        'tokens_per_second': tokens_per_second,
                    }
                }
            )
            
        except httpx.ConnectError:
            raise OllamaError(
                f"Could not reach Ollama server at {self.base_url}",
                model=self.model,
                base_url=self.base_url,
                suggestion="Ensure Ollama is running.\n"
                           "Install: https://ollama.ai\n"
                           "Start: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            # Check for context window errors
            error_text = e.response.text
            if 'context' in error_text.lower() or 'too large' in error_text.lower():
                raise OllamaError(
                    f"Input is too large for model '{self.model}' context window.",
                    model=self.model,
                    base_url=self.base_url,
                    suggestion="Reduce prompt length or use a model with a larger context window.\n"
                               f"Current context setting: {self.num_ctx} tokens"
                )
            
            raise OllamaError(
                f"Ollama API error: {e.response.status_code}",
                model=self.model,
                base_url=self.base_url,
                suggestion="Check Ollama logs for details.",
                original_error=e
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise OllamaError(
                "Network error communicating with Ollama.",
                model=self.model,
                base_url=self.base_url,
                original_error=e
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise OllamaError(
                "Unexpected error during generation.",
                model=self.model,
                base_url=self.base_url,
                original_error=e
            )
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> AsyncIterator[ProviderResponse]:
        """
        Generate streaming text completion using Ollama.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Yields:
            ProviderResponse chunks with partial content
            
        Raises:
            OllamaError: If streaming fails with helpful context
        """
        await self._ensure_ready()
        
        # Build request payload for streaming
        payload = self._build_request_payload(messages, stream=True, **kwargs)
        
        start_time = time.time()
        chunk_count = 0
        total_tokens = 0
        
        try:
            client = await self._get_http_client()
            
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        chunk_count += 1
                        
                        content = chunk_data.get('message', {}).get('content', '')
                        done = chunk_data.get('done', False)
                        
                        # Track tokens
                        if done:
                            prompt_tokens = chunk_data.get('prompt_eval_count', 0)
                            completion_tokens = chunk_data.get('eval_count', 0)
                            total_tokens = prompt_tokens + completion_tokens
                        
                        if content or done:
                            yield ProviderResponse(
                                model=chunk_data.get('model', self.model),
                                output_text=content,
                                raw=chunk_data,
                                metadata={
                                    'created_at': chunk_data.get('created_at'),
                                    'done': done,
                                    'provider': 'ollama',
                                    'chunk_number': chunk_count,
                                    'tokens': {
                                        'prompt': chunk_data.get('prompt_eval_count', 0),
                                        'completion': chunk_data.get('eval_count', 0),
                                    } if done else {}
                                }
                            )
                            
                        if done:
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama stream chunk: {line}")
                        continue
            
            # Record streaming metrics
            duration = time.time() - start_time
            tokens_per_second = total_tokens / duration if duration > 0 and total_tokens > 0 else 0
            
            record_metric("ollama.stream.duration", duration, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.stream.chunks", chunk_count, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.stream.tokens", total_tokens, {
                "provider": self.name,
                "model": self.model
            })
            record_metric("ollama.stream.throughput", tokens_per_second, {
                "provider": self.name,
                "model": self.model
            })
            
            logger.info(
                f"Ollama stream completed: {chunk_count} chunks, {total_tokens} tokens "
                f"in {duration:.2f}s ({tokens_per_second:.1f} tok/s)"
            )
            
        except httpx.ConnectError:
            raise OllamaError(
                f"Could not reach Ollama server at {self.base_url}",
                model=self.model,
                base_url=self.base_url,
                suggestion="Ensure Ollama is running.\n"
                           "Install: https://ollama.ai\n"
                           "Start: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            if 'context' in error_text.lower() or 'too large' in error_text.lower():
                raise OllamaError(
                    f"Input is too large for model '{self.model}' context window.",
                    model=self.model,
                    base_url=self.base_url,
                    suggestion="Reduce prompt length or use a model with a larger context window.\n"
                               f"Current context setting: {self.num_ctx} tokens"
                )
            
            raise OllamaError(
                f"Ollama streaming API error: {e.response.status_code}",
                model=self.model,
                base_url=self.base_url,
                original_error=e
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming HTTP error: {e}")
            raise OllamaError(
                "Network error during streaming.",
                model=self.model,
                base_url=self.base_url,
                original_error=e
            )
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise OllamaError(
                "Unexpected error during streaming.",
                model=self.model,
                base_url=self.base_url,
                original_error=e
            )
    
    def _build_request_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for Ollama API."""
        # Convert messages to Ollama format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        # Build options
        options = {}
        
        # Add generation parameters
        options['temperature'] = kwargs.get('temperature', self.temperature)
        options['top_k'] = kwargs.get('top_k', self.top_k)
        options['top_p'] = kwargs.get('top_p', self.top_p)
        options['repeat_penalty'] = kwargs.get('repeat_penalty', self.repeat_penalty)
        options['num_ctx'] = kwargs.get('num_ctx', self.num_ctx)
        
        # Add deployment config options
        if 'num_gpu' in self.deployment_config:
            options['num_gpu'] = self.deployment_config['num_gpu']
        if 'num_thread' in self.deployment_config:
            options['num_thread'] = self.deployment_config['num_thread']
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "options": options,
            "stream": stream,
            "keep_alive": self.keep_alive,
        }
        
        return payload
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.debug("Closed HTTP client")
        
        # Clear cache
        await self._cache.clear()
        
        # Optionally stop managed server
        if self.auto_start_server:
            try:
                await self._server_manager.stop_server()
            except Exception as e:
                logger.warning(f"Failed to stop Ollama server: {e}")
    
    async def health_check(self, *, force: bool = False) -> Dict[str, Any]:
        """Check provider health with detailed diagnostics.
        
        Args:
            force: Force fresh health check (ignore throttling)
            
        Returns:
            Comprehensive health status dictionary
        """
        try:
            health_data = await self._server_manager.health_check(force=force)
            models = await self._model_manager.list_models()
            
            model_available = any(m.get('name') == self.model for m in models)
            
            return {
                'status': 'healthy' if model_available else 'model_missing',
                'provider': 'ollama',
                'model': self.model,
                'model_available': model_available,
                'base_url': self.base_url,
                'server_health': health_data,
                'available_models': [m.get('name') for m in models],
                'configuration': {
                    'auto_pull': self.auto_pull_model,
                    'auto_start': self.auto_start_server,
                    'context_window': self.num_ctx,
                    'temperature': self.temperature,
                }
            }
        except OllamaError as e:
            return {
                'status': 'unhealthy',
                'provider': 'ollama',
                'model': self.model,
                'base_url': self.base_url,
                'error': str(e),
                'suggestion': e.suggestion,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': 'ollama',
                'model': self.model,
                'base_url': self.base_url,
                'error': str(e),
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models.
        
        Returns:
            List of model metadata dictionaries
            
        Raises:
            OllamaError: If server is unreachable
        """
        return await self._model_manager.list_models()
    
    async def pull_model(self, model_name: Optional[str] = None, *, pull_timeout: int = 600) -> None:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Model name to pull (defaults to configured model)
            pull_timeout: Maximum time to wait for pull in seconds
            
        Raises:
            OllamaError: If pull fails
        """
        target_model = model_name or self.model
        await self._model_manager.ensure_model_available(
            target_model,
            auto_pull=True,
            pull_timeout=pull_timeout
        )
    
    async def delete_model(self, model_name: Optional[str] = None) -> None:
        """Delete a model from Ollama.
        
        Args:
            model_name: Model name to delete (defaults to configured model)
            
        Raises:
            OllamaError: If deletion fails
        """
        target_model = model_name or self.model
        await self._model_manager.delete_model(target_model)


# Register the provider
register_provider_class("ollama", OllamaProvider)


# =============================================================================
# Editor Integration Hooks
# =============================================================================

class OllamaEditorTools:
    """Tools and utilities for editor/IDE integrations (VSCode, etc.).
    
    This class provides a stable API for external tools to query Ollama
    status and capabilities without instantiating a full provider.
    """
    
    @staticmethod
    async def check_status(base_url: Optional[str] = None) -> Dict[str, Any]:
        """Check if Ollama server is reachable and get basic status.
        
        Args:
            base_url: Ollama server URL (defaults to env or standard URL)
            
        Returns:
            Status dictionary with:
            - reachable: bool
            - base_url: str
            - models_count: int (if reachable)
            - error: str (if not reachable)
        
        Example:
            >>> status = await OllamaEditorTools.check_status()
            >>> if status['reachable']:
            ...     print(f"Ollama is running with {status['models_count']} models")
        """
        url = base_url or os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_URL)
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = data.get('models', [])
                
                return {
                    'reachable': True,
                    'base_url': url,
                    'models_count': len(models),
                    'models': [m.get('name') for m in models],
                }
        except httpx.ConnectError:
            return {
                'reachable': False,
                'base_url': url,
                'error': 'Cannot connect to Ollama server',
                'suggestion': 'Ensure Ollama is installed and running (ollama serve)',
            }
        except Exception as e:
            return {
                'reachable': False,
                'base_url': url,
                'error': str(e),
            }
    
    @staticmethod
    async def list_available_models(base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available models for autocomplete/dropdown.
        
        Args:
            base_url: Ollama server URL (defaults to env or standard URL)
            
        Returns:
            List of model dictionaries with name, size, and metadata
            
        Example:
            >>> models = await OllamaEditorTools.list_available_models()
            >>> for model in models:
            ...     print(f"{model['name']} - {model.get('size', 'unknown size')}")
        """
        url = base_url or os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_URL)
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = data.get('models', [])
                
                # Format for editor consumption
                return [{
                    'name': m.get('name', 'unknown'),
                    'size': m.get('size', 0),
                    'modified_at': m.get('modified_at'),
                    'digest': m.get('digest'),
                } for m in models]
                
        except Exception as e:
            logger.warning(f"Failed to list models for editor: {e}")
            return []
    
    @staticmethod
    def get_default_model() -> str:
        """Get the recommended default model for new projects.
        
        Returns:
            Default model name
        """
        return "llama3:8b"
    
    @staticmethod
    def get_supported_capabilities() -> Dict[str, bool]:
        """Get capabilities supported by Ollama provider.
        
        Returns:
            Dictionary of capability flags for IDE features
        """
        return {
            'chat': True,
            'completion': True,
            'streaming': True,
            'embeddings': False,  # Not yet implemented
            'function_calling': False,  # Not supported by Ollama
            'vision': False,  # Model-dependent
            'auto_pull': True,
            'local_deployment': True,
        }
    
    @staticmethod
    def validate_model_name(model_name: str) -> Dict[str, Any]:
        """Validate model name format for linting/diagnostics.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            Validation result with is_valid and message
        
        Example:
            >>> result = OllamaEditorTools.validate_model_name("llama3:8b")
            >>> if not result['is_valid']:
            ...     print(result['message'])
        """
        if not model_name:
            return {
                'is_valid': False,
                'message': 'Model name cannot be empty',
            }
        
        # Basic format check (name:tag or just name)
        if ':' in model_name:
            parts = model_name.split(':')
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return {
                    'is_valid': False,
                    'message': 'Invalid model format. Expected: model_name:tag',
                }
        
        # Check for invalid characters
        if any(c in model_name for c in [' ', '\t', '\n', '\r']):
            return {
                'is_valid': False,
                'message': 'Model name cannot contain whitespace',
            }
        
        return {
            'is_valid': True,
            'message': 'Valid model name',
        }


__all__ = [
    "OllamaProvider",
    "OllamaModelManager",
    "OllamaServerManager",
    "OllamaError",
    "ModelAvailabilityCache",
    "OllamaEditorTools",
    "OLLAMA_BASE_URL_ENV",
    "DEFAULT_OLLAMA_URL",
]