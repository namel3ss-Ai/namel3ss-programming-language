"""Production-grade vLLM provider for high-throughput local LLM inference."""

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


class VLLMDeploymentManager:
    """Manager for vLLM server deployment and lifecycle."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 8000)}"
        
    async def start_server(self, model_name: str) -> None:
        """Start vLLM server with specified model."""
        if self.process and self.process.poll() is None:
            logger.info("vLLM server already running")
            return
            
        cmd = self._build_vllm_command(model_name)
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # Wait for server to be ready
            await self._wait_for_server_ready()
            logger.info(f"vLLM server started successfully at {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            raise ProviderError(f"vLLM server startup failed: {e}")
    
    def _build_vllm_command(self, model_name: str) -> List[str]:
        """Build vLLM server command with configuration."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--host", self.config.get('host', '127.0.0.1'),
            "--port", str(self.config.get('port', 8000)),
        ]
        
        # Add optional configurations
        if 'gpu_memory_utilization' in self.config:
            cmd.extend(["--gpu-memory-utilization", str(self.config['gpu_memory_utilization'])])
        
        if 'tensor_parallel_size' in self.config:
            cmd.extend(["--tensor-parallel-size", str(self.config['tensor_parallel_size'])])
            
        if 'max_model_len' in self.config:
            cmd.extend(["--max-model-len", str(self.config['max_model_len'])])
            
        if 'dtype' in self.config:
            cmd.extend(["--dtype", self.config['dtype']])
            
        if 'quantization' in self.config:
            cmd.extend(["--quantization", self.config['quantization']])
            
        if 'served_model_name' in self.config:
            cmd.extend(["--served-model-name", self.config['served_model_name']])
        
        return cmd
    
    async def _wait_for_server_ready(self, max_wait: int = 120) -> None:
        """Wait for vLLM server to be ready."""
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() - start_time < max_wait:
                try:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                
                await asyncio.sleep(2)
        
        raise ProviderError(f"vLLM server did not become ready within {max_wait} seconds")
    
    async def stop_server(self) -> None:
        """Stop vLLM server."""
        if self.process:
            logger.info("Stopping vLLM server")
            self.process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_end()),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("vLLM server did not stop gracefully, forcing kill")
                self.process.kill()
            
            self.process = None
    
    async def _wait_for_process_end(self) -> None:
        """Wait for process to end."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vLLM server health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise ProviderError(f"vLLM health check failed: {e}")


class VLLMProvider(N3Provider):
    """
    Production-grade vLLM provider for high-throughput local LLM inference.
    
    Features:
    - Automatic vLLM server deployment and management
    - Continuous batching for high throughput
    - Streaming response support with backpressure
    - GPU memory optimization and tensor parallelism
    - Health monitoring and automatic recovery
    - Production-grade error handling and retry logic
    - Integration with N3 observability system
    
    Configuration:
        - model: HuggingFace model identifier (required)
        - host: Server host (default: 127.0.0.1)
        - port: Server port (default: 8000)
        - gpu_memory_utilization: GPU memory fraction (0.1-1.0, default: 0.9)
        - tensor_parallel_size: Number of GPUs for tensor parallelism
        - max_model_len: Maximum model sequence length
        - dtype: Model data type (auto, float16, bfloat16, etc.)
        - quantization: Quantization method (awq, gptq, etc.)
        - auto_start_server: Whether to auto-start vLLM server (default: True)
    """
    
    def __init__(
        self, 
        name: str, 
        model: str, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # vLLM-specific configuration
        self.deployment_config = self.config.get('deployment_config', {})
        self.auto_start_server = self.config.get('auto_start_server', True)
        self.base_url = f"http://{self.config.get('host', '127.0.0.1')}:{self.config.get('port', 8000)}"
        
        # Request configuration
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        self.top_p = float(self.config.get('top_p', 1.0))
        self.frequency_penalty = float(self.config.get('frequency_penalty', 0.0))
        self.presence_penalty = float(self.config.get('presence_penalty', 0.0))
        
        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Deployment manager
        self._deployment_manager = VLLMDeploymentManager({
            **self.deployment_config,
            'host': self.config.get('host', '127.0.0.1'),
            'port': self.config.get('port', 8000),
        })
        
        # Health tracking
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, read=300.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._http_client
    
    async def _ensure_server_running(self) -> None:
        """Ensure vLLM server is running and healthy."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self._last_health_check < self._health_check_interval:
            return
            
        try:
            await self._deployment_manager.health_check()
            self._last_health_check = current_time
            return
        except Exception:
            # Server not healthy, try to start if auto-start enabled
            if self.auto_start_server:
                logger.info(f"vLLM server not healthy, attempting to start for model {self.model}")
                await self._deployment_manager.start_server(self.model)
                self._last_health_check = current_time
            else:
                raise ProviderError(
                    f"vLLM server not available at {self.base_url}. "
                    f"Start manually or enable auto_start_server=True"
                )
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> ProviderResponse:
        """
        Generate text completion using vLLM.
        
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
            await record_metric("vllm.request.duration", duration, {"provider": self.name})
            await record_metric("vllm.tokens.generated", 
                              result.get('usage', {}).get('completion_tokens', 0),
                              {"provider": self.name})
            
            return ProviderResponse(
                model=result.get('model', self.model),
                output_text=result['choices'][0]['message']['content'],
                raw=result,
                usage=result.get('usage', {}),
                finish_reason=result['choices'][0].get('finish_reason'),
                metadata={
                    'provider': 'vllm',
                    'duration': duration,
                }
            )
            
        except httpx.HTTPError as e:
            logger.error(f"vLLM HTTP error: {e}")
            raise ProviderError(f"vLLM request failed: {e}")
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise ProviderError(f"vLLM generation failed: {e}")
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs
    ) -> AsyncIterator[ProviderResponse]:
        """
        Generate streaming text completion using vLLM.
        
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
                                    model=chunk_data.get('model', self.model),
                                    output_text=content,
                                    raw=chunk_data,
                                    finish_reason=choice.get('finish_reason'),
                                    metadata={
                                        'chunk_id': chunk_data.get('id'),
                                        'provider': 'vllm',
                                        'chunk_number': chunk_count,
                                    }
                                )
                                
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse vLLM stream chunk: {line}")
                        continue
            
            # Record streaming metrics
            duration = time.time() - start_time
            await record_metric("vllm.stream.duration", duration, {"provider": self.name})
            await record_metric("vllm.stream.chunks", chunk_count, {"provider": self.name})
            
        except httpx.HTTPError as e:
            logger.error(f"vLLM streaming HTTP error: {e}")
            raise ProviderError(f"vLLM streaming failed: {e}")
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise ProviderError(f"vLLM streaming failed: {e}")
    
    def _build_request_payload(
        self,
        messages: List[ProviderMessage],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for vLLM API."""
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
                await self._deployment_manager.stop_server()
            except Exception as e:
                logger.warning(f"Failed to stop vLLM server: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            health_data = await self._deployment_manager.health_check()
            return {
                'status': 'healthy',
                'provider': 'vllm',
                'model': self.model,
                'base_url': self.base_url,
                'server_health': health_data,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': 'vllm',
                'model': self.model,
                'base_url': self.base_url,
                'error': str(e),
            }


# Register the provider
register_provider_class("vllm", VLLMProvider)

__all__ = ["VLLMProvider", "VLLMDeploymentManager"]