"""Production-grade LocalAI/LM Studio provider for multi-format local model support."""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx

from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric
from ..base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from ..factory import register_provider_class

logger = get_logger(__name__)


class LocalAIServerManager:
    """Manager for LocalAI server deployment and lifecycle."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 8080)
        self.base_url = f"http://{self.host}:{self.port}"
        
    async def start_server(self) -> None:
        """Start LocalAI server if not running."""
        if await self._is_server_running():
            logger.info("LocalAI server already running")
            return
            
        logger.info("Starting LocalAI server")
        
        try:
            cmd = self._build_localai_command()
            
            # Set environment variables
            env = os.environ.copy()
            env.update(self.config.get('env', {}))
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Wait for server to be ready
            await self._wait_for_server_ready()
            logger.info(f"LocalAI server started at {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to start LocalAI server: {e}")
            raise ProviderError(f"LocalAI server startup failed: {e}")
    
    def _build_localai_command(self) -> List[str]:
        """Build LocalAI server command."""
        # Check if running in Docker or local binary
        if self.config.get('use_docker', False):
            return self._build_docker_command()
        else:
            return self._build_binary_command()
    
    def _build_docker_command(self) -> List[str]:
        """Build Docker command for LocalAI."""
        cmd = [
            "docker", "run", "-d",
            "--name", self.config.get('container_name', 'localai'),
            "-p", f"{self.port}:8080",
        ]
        
        # Add GPU support if configured
        if self.config.get('gpu', False):
            cmd.extend(["--gpus", "all"])
        
        # Add model volume mount
        models_path = self.config.get('models_path', './models')
        cmd.extend(["-v", f"{models_path}:/models"])
        
        # Add environment variables
        env_vars = self.config.get('env', {})
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Add LocalAI image
        image = self.config.get('image', 'quay.io/go-skynet/local-ai:latest')
        cmd.append(image)
        
        return cmd
    
    def _build_binary_command(self) -> List[str]:
        """Build binary command for LocalAI."""
        binary_path = self.config.get('binary_path', 'local-ai')
        
        cmd = [
            binary_path,
            "--address", f"{self.host}:{self.port}",
        ]
        
        # Add configuration options
        if 'models_path' in self.config:
            cmd.extend(["--models-path", self.config['models_path']])
        
        if 'config_file' in self.config:
            cmd.extend(["--config-file", self.config['config_file']])
        
        if 'debug' in self.config and self.config['debug']:
            cmd.append("--debug")
        
        return cmd
    
    async def _is_server_running(self) -> bool:
        """Check if LocalAI server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                return response.status_code == 200
        except Exception:
            return False
    
    async def _wait_for_server_ready(self, max_wait: int = 120) -> None:
        """Wait for LocalAI server to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self._is_server_running():
                return
            await asyncio.sleep(2)
        
        raise ProviderError(f"LocalAI server did not start within {max_wait} seconds")
    
    async def stop_server(self) -> None:
        """Stop LocalAI server."""
        if self.config.get('use_docker', False):
            await self._stop_docker_container()
        elif self.process:
            logger.info("Stopping LocalAI server")
            self.process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_end()),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("LocalAI server did not stop gracefully, forcing kill")
                self.process.kill()
            
            self.process = None
    
    async def _stop_docker_container(self) -> None:
        """Stop LocalAI Docker container."""
        container_name = self.config.get('container_name', 'localai')
        try:
            subprocess.run(["docker", "stop", container_name], check=True)
            subprocess.run(["docker", "rm", container_name], check=True)
            logger.info(f"Stopped LocalAI Docker container: {container_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop LocalAI Docker container: {e}")
    
    async def _wait_for_process_end(self) -> None:
        """Wait for process to end."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LocalAI server health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                
                data = response.json()
                return {
                    'status': 'healthy',
                    'models': data.get('data', []),
                    'base_url': self.base_url,
                }
        except Exception as e:
            raise ProviderError(f"LocalAI health check failed: {e}")


class LocalAIProvider(N3Provider):
    """
    Production-grade LocalAI/LM Studio provider for multi-format local models.
    
    Features:
    - Support for GGML, GGUF, and other quantized formats
    - Multiple backend engines (llama.cpp, gpt4all, etc.)
    - Automatic server management (binary or Docker)
    - OpenAI-compatible API interface
    - Streaming response support
    - GPU acceleration support
    - Flexible model configuration
    - Integration with N3 observability system
    
    Configuration:
        - model: Model identifier for LocalAI
        - host: Server host (default: 127.0.0.1)
        - port: Server port (default: 8080)
        - auto_start_server: Whether to auto-start LocalAI server (default: True)
        - use_docker: Whether to run LocalAI in Docker (default: False)
        - models_path: Path to model files
        - config_file: LocalAI configuration file path
        - binary_path: Path to LocalAI binary
        
    Deployment Configuration:
        - backend: Backend engine (llama, gpt4all, etc.)
        - f16: Use float16 precision
        - threads: Number of CPU threads
        - context_length: Context window size
        - batch_size: Batch size for processing
        - gpu_layers: Number of layers to run on GPU
        - mmap: Use memory mapping
        - mlock: Lock model in memory
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # LocalAI-specific configuration
        self.deployment_config = self.config.get('deployment_config', {})
        self.auto_start_server = self.config.get('auto_start_server', True)
        
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 8080)
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        self.top_p = float(self.config.get('top_p', 1.0))
        self.top_k = self.config.get('top_k', 40)
        self.frequency_penalty = float(self.config.get('frequency_penalty', 0.0))
        self.presence_penalty = float(self.config.get('presence_penalty', 0.0))
        
        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Server manager
        self._server_manager = LocalAIServerManager({
            **self.deployment_config,
            **{k: v for k, v in self.config.items() if k in [
                'host', 'port', 'use_docker', 'models_path', 'config_file', 
                'binary_path', 'container_name', 'image', 'gpu', 'env'
            ]}
        })
        
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
    
    async def _ensure_server_running(self) -> None:
        """Ensure LocalAI server is running and healthy."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self._last_health_check < self._health_check_interval:
            return
            
        try:
            await self._server_manager.health_check()
            self._last_health_check = current_time
            return
        except Exception:
            # Server not healthy, try to start if auto-start enabled
            if self.auto_start_server:
                logger.info(f"LocalAI server not healthy, attempting to start")
                await self._server_manager.start_server()
                self._last_health_check = current_time
            else:
                raise ProviderError(
                    f"LocalAI server not available at {self.base_url}. "
                    f"Start manually or enable auto_start_server=True"
                )
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> ProviderResponse:
        """
        Generate text completion using LocalAI.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            ProviderResponse with generated text
        """
        await self._ensure_server_running()
        
        # Build request payload
        payload = self._build_request_payload(messages, stream=False, **kwargs)
        
        start_time = time.time()
        
        try:
            client = await self._get_http_client()
            response = await client.post(
                urljoin(self.base_url, "/v1/chat/completions"),
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Record metrics
            duration = time.time() - start_time
            await record_metric("localai.request.duration", duration, {"provider": self.name})
            await record_metric("localai.tokens.generated", 
                              result.get('usage', {}).get('completion_tokens', 0),
                              {"provider": self.name})
            
            return ProviderResponse(
                model=result.get('model', self.model),
                output_text=result['choices'][0]['message']['content'],
                raw=result,
                usage=result.get('usage', {}),
                finish_reason=result['choices'][0].get('finish_reason'),
                metadata={
                    'provider': 'local_ai',
                    'duration': duration,
                }
            )
            
        except httpx.HTTPError as e:
            logger.error(f"LocalAI HTTP error: {e}")
            raise ProviderError(f"LocalAI request failed: {e}")
        except Exception as e:
            logger.error(f"LocalAI generation error: {e}")
            raise ProviderError(f"LocalAI generation failed: {e}")
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> AsyncIterator[ProviderResponse]:
        """
        Generate streaming text completion using LocalAI.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Yields:
            ProviderResponse chunks with partial content
        """
        await self._ensure_server_running()
        
        # Build request payload for streaming
        payload = self._build_request_payload(messages, stream=True, **kwargs)
        
        start_time = time.time()
        chunk_count = 0
        
        try:
            client = await self._get_http_client()
            
            async with client.stream(
                "POST",
                urljoin(self.base_url, "/v1/chat/completions"),
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                        
                    if line.startswith("data: "):
                        line = line[6:]
                        
                    try:
                        chunk_data = json.loads(line)
                        chunk_count += 1
                        
                        if 'choices' in chunk_data and chunk_data['choices']:
                            choice = chunk_data['choices'][0]
                            delta = choice.get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                yield ProviderResponse(
                                    content=content,
                                    metadata={
                                        'model': chunk_data.get('model', self.model),
                                        'chunk_id': chunk_data.get('id'),
                                        'finish_reason': choice.get('finish_reason'),
                                        'provider': 'local_ai',
                                        'chunk_number': chunk_count,
                                    }
                                )
                                
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse LocalAI stream chunk: {line}")
                        continue
            
            # Record streaming metrics
            duration = time.time() - start_time
            await record_metric("localai.stream.duration", duration, {"provider": self.name})
            await record_metric("localai.stream.chunks", chunk_count, {"provider": self.name})
            
        except httpx.HTTPError as e:
            logger.error(f"LocalAI streaming HTTP error: {e}")
            raise ProviderError(f"LocalAI streaming failed: {e}")
        except Exception as e:
            logger.error(f"LocalAI streaming error: {e}")
            raise ProviderError(f"LocalAI streaming failed: {e}")
    
    def _build_request_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for LocalAI API."""
        # Convert messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "top_p": kwargs.get('top_p', self.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.presence_penalty),
            "stream": stream,
        }
        
        # Add top_k if specified
        if self.top_k is not None:
            payload["top_k"] = kwargs.get('top_k', self.top_k)
        
        # Add stop sequences if provided
        stop_sequences = kwargs.get('stop_sequences', self.config.get('stop_sequences'))
        if stop_sequences:
            payload["stop"] = stop_sequences
        
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
                logger.warning(f"Failed to stop LocalAI server: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            health_data = await self._server_manager.health_check()
            return {
                'status': 'healthy',
                'provider': 'local_ai',
                'model': self.model,
                'base_url': self.base_url,
                'server_health': health_data,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': 'local_ai',
                'model': self.model,
                'base_url': self.base_url,
                'error': str(e),
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        await self._ensure_server_running()
        
        try:
            client = await self._get_http_client()
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except Exception as e:
            logger.error(f"Failed to list LocalAI models: {e}")
            raise ProviderError(f"Failed to list LocalAI models: {e}")


# Register the provider
register_provider_class("local_ai", LocalAIProvider)
register_provider_class("lm_studio", LocalAIProvider)  # LM Studio uses same interface

__all__ = ["LocalAIProvider", "LocalAIServerManager"]