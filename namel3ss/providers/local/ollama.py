"""Production-grade Ollama provider for local model deployment and management."""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union
from urllib.parse import urljoin

import httpx

from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric
from ..base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from ..factory import register_provider_class

logger = get_logger(__name__)


class OllamaModelManager:
    """Manager for Ollama model pulling and lifecycle."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._pulled_models: Set[str] = set()
        
    async def ensure_model_available(self, model_name: str) -> None:
        """Ensure model is pulled and available."""
        if model_name in self._pulled_models:
            return
            
        # Check if model is already available
        if await self._is_model_available(model_name):
            self._pulled_models.add(model_name)
            return
        
        # Pull the model
        logger.info(f"Pulling Ollama model: {model_name}")
        await self._pull_model(model_name)
        self._pulled_models.add(model_name)
    
    async def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available locally."""
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
                
        except Exception as e:
            logger.warning(f"Failed to check Ollama model availability: {e}")
            return False
    
    async def _pull_model(self, model_name: str, timeout: int = 300) -> None:
        """Pull model from Ollama registry."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                status_data = json.loads(line)
                                status = status_data.get('status', '')
                                if 'pulling' in status.lower() or 'downloading' in status.lower():
                                    logger.info(f"Ollama: {status}")
                                elif status_data.get('status') == 'success':
                                    logger.info(f"Successfully pulled Ollama model: {model_name}")
                                    return
                            except json.JSONDecodeError:
                                continue
                
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            raise ProviderError(f"Ollama model pull failed: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                return data.get('models', [])
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise ProviderError(f"Failed to list Ollama models: {e}")
    
    async def delete_model(self, model_name: str) -> None:
        """Delete a model from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/delete",
                    json={"name": model_name}
                )
                response.raise_for_status()
                
                self._pulled_models.discard(model_name)
                logger.info(f"Deleted Ollama model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to delete Ollama model {model_name}: {e}")
            raise ProviderError(f"Ollama model deletion failed: {e}")


class OllamaServerManager:
    """Manager for Ollama server deployment and lifecycle."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 11434)
        self.base_url = f"http://{self.host}:{self.port}"
        
    async def start_server(self) -> None:
        """Start Ollama server if not running."""
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
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            raise ProviderError(f"Ollama server startup failed: {e}")
    
    async def _is_server_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def _wait_for_server_ready(self, max_wait: int = 60) -> None:
        """Wait for Ollama server to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self._is_server_running():
                return
            await asyncio.sleep(2)
        
        raise ProviderError(f"Ollama server did not start within {max_wait} seconds")
    
    async def stop_server(self) -> None:
        """Stop Ollama server."""
        if self.process:
            logger.info("Stopping Ollama server")
            self.process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_end()),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Ollama server did not stop gracefully, forcing kill")
                self.process.kill()
            
            self.process = None
    
    async def _wait_for_process_end(self) -> None:
        """Wait for process to end."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama server health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                return {
                    'status': 'healthy',
                    'models': data.get('models', []),
                    'base_url': self.base_url,
                }
        except Exception as e:
            raise ProviderError(f"Ollama health check failed: {e}")


class OllamaProvider(N3Provider):
    """
    Production-grade Ollama provider for local model deployment.
    
    Features:
    - Automatic Ollama server management
    - Model pulling and lifecycle management
    - Streaming response support
    - GPU acceleration and optimization
    - Health monitoring and recovery
    - Integration with N3 observability system
    - Support for all Ollama-compatible models
    
    Configuration:
        - model: Ollama model name (e.g., "llama3", "mistral", "codellama")
        - host: Server host (default: 127.0.0.1)
        - port: Server port (default: 11434)
        - auto_pull_model: Whether to auto-pull missing models (default: True)
        - auto_start_server: Whether to auto-start Ollama server (default: True)
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
        self.auto_start_server = self.config.get('auto_start_server', True)
        
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 11434)
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.top_k = self.config.get('top_k', 40)
        self.top_p = float(self.config.get('top_p', 0.9))
        self.repeat_penalty = float(self.config.get('repeat_penalty', 1.1))
        self.num_ctx = self.config.get('num_ctx', 2048)
        self.keep_alive = self.config.get('keep_alive', "5m")
        
        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Managers
        self._server_manager = OllamaServerManager({
            **self.deployment_config,
            'host': self.host,
            'port': self.port,
        })
        self._model_manager = OllamaModelManager(self.base_url)
        
        # Health tracking
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, read=300.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
            )
        return self._http_client
    
    async def _ensure_ready(self) -> None:
        """Ensure Ollama server and model are ready."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        # Ensure server is running
        if self.auto_start_server:
            await self._server_manager.start_server()
        
        # Ensure model is available
        if self.auto_pull_model:
            await self._model_manager.ensure_model_available(self.model)
        
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
            
            # Record metrics
            duration = time.time() - start_time
            await record_metric("ollama.request.duration", duration, {"provider": self.name})
            
            return ProviderResponse(
                model=result.get('model', self.model),
                output_text=result['message']['content'],
                raw=result,
                metadata={
                    'created_at': result.get('created_at'),
                    'done': result.get('done', True),
                    'provider': 'ollama',
                    'duration': duration,
                }
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise ProviderError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise ProviderError(f"Ollama generation failed: {e}")
    
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
        """
        await self._ensure_ready()
        
        # Build request payload for streaming
        payload = self._build_request_payload(messages, stream=True, **kwargs)
        
        start_time = time.time()
        chunk_count = 0
        
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
                                }
                            )
                            
                        if done:
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama stream chunk: {line}")
                        continue
            
            # Record streaming metrics
            duration = time.time() - start_time
            await record_metric("ollama.stream.duration", duration, {"provider": self.name})
            await record_metric("ollama.stream.chunks", chunk_count, {"provider": self.name})
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming HTTP error: {e}")
            raise ProviderError(f"Ollama streaming failed: {e}")
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise ProviderError(f"Ollama streaming failed: {e}")
    
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
        
        # Optionally stop managed server
        if self.auto_start_server:
            try:
                await self._server_manager.stop_server()
            except Exception as e:
                logger.warning(f"Failed to stop Ollama server: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            health_data = await self._server_manager.health_check()
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
        """List available models."""
        await self._ensure_ready()
        return await self._model_manager.list_models()
    
    async def pull_model(self, model_name: Optional[str] = None) -> None:
        """Pull a model."""
        await self._ensure_ready()
        target_model = model_name or self.model
        await self._model_manager.ensure_model_available(target_model)
    
    async def delete_model(self, model_name: Optional[str] = None) -> None:
        """Delete a model."""
        await self._ensure_ready()
        target_model = model_name or self.model
        await self._model_manager.delete_model(target_model)


# Register the provider
register_provider_class("ollama", OllamaProvider)

__all__ = ["OllamaProvider", "OllamaModelManager", "OllamaServerManager"]