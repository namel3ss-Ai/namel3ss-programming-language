"""Google (Vertex AI / Gemini) provider implementation for N3Provider.

Production-grade Google AI integration with async support.
"""

import os
from typing import Any, AsyncIterable, Dict, List, Optional

from .base import N3Provider, ProviderMessage, ProviderResponse, ProviderError
from .factory import register_provider_class


class GoogleProvider(N3Provider):
    """
    Google provider implementation.
    
    Supports Vertex AI (PaLM, Gemini) and Gemini API models.
    
    Configuration (via NAMEL3SS_PROVIDER_GOOGLE_* or DSL config):
        - project: GCP project ID (required for Vertex AI)
        - location: GCP location (default: us-central1)
        - api_key: API key (for Gemini API)
        - credentials_path: Path to service account JSON (optional)
        - timeout: Request timeout in seconds (default: 60)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 1024)
        - top_p: Nucleus sampling parameter
        - top_k: Top-k sampling parameter
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, model, config)
        
        # Configuration
        self.project = self.config.get('project') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location = self.config.get('location', 'us-central1')
        self.api_key = self.config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.credentials_path = self.config.get('credentials_path') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.timeout = float(self.config.get('timeout', 60))
        
        # Check if we have either project (Vertex) or API key (Gemini)
        if not self.project and not self.api_key:
            raise ProviderError(
                f"Google configuration not found for provider '{name}'. "
                f"Set NAMEL3SS_PROVIDER_GOOGLE_PROJECT or NAMEL3SS_PROVIDER_GOOGLE_API_KEY, "
                f"or provide 'project' or 'api_key' in config."
            )
        
        # Generation parameters
        self.temperature = float(self.config.get('temperature', 0.7))
        self.max_tokens = int(self.config.get('max_tokens', 1024))
        
        # HTTP client (initialized lazily)
        self._http_client = None
        self._use_vertex = self.project is not None
    
    def _get_http_client(self):
        """Get or create async HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
            except ImportError:
                raise ProviderError(
                    "httpx is required for Google provider. Install it with: pip install httpx"
                )
        return self._http_client
    
    async def _get_access_token(self) -> str:
        """Get GCP access token for Vertex AI."""
        if not self._use_vertex:
            return None
        
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            
            credentials, _ = default()
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            raise ProviderError(f"Failed to get GCP access token: {e}") from e
    
    def _build_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Build request headers."""
        headers = {'Content-Type': 'application/json'}
        
        if self._use_vertex and access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        elif self.api_key:
            # Gemini API uses key in URL parameter
            pass
        
        return headers
    
    def _convert_messages(self, messages: List[ProviderMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Google format."""
        converted = []
        
        for msg in messages:
            role = msg.role
            # Map roles to Google format
            if role == 'assistant':
                role = 'model'
            elif role == 'system':
                # Google doesn't have explicit system role, prepend to first user message
                # For now, treat as user message
                role = 'user'
            
            converted.append({
                'role': role,
                'parts': [{'text': msg.content}],
            })
        
        return converted
    
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Returns:
            ProviderResponse with generated text and metadata
        
        Raises:
            ProviderError: If the API call fails
        """
        client = self._get_http_client()
        
        # Build URL based on Vertex AI or Gemini API
        if self._use_vertex:
            access_token = await self._get_access_token()
            headers = self._build_headers(access_token)
            url = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project}/locations/{self.location}/"
                f"publishers/google/models/{self.model}:generateContent"
            )
        else:
            headers = self._build_headers()
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"
        
        # Build payload
        converted_messages = self._convert_messages(messages)
        payload = {
            'contents': converted_messages,
            'generationConfig': {
                'temperature': kwargs.get('temperature', self.temperature),
                'maxOutputTokens': kwargs.get('max_tokens', self.max_tokens),
            },
        }
        
        # Optional parameters
        if 'top_p' in kwargs:
            payload['generationConfig']['topP'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['generationConfig']['topK'] = kwargs['top_k']
        
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            candidates = data.get('candidates', [])
            if not candidates:
                raise ProviderError(
                    f"Google API returned no candidates for provider '{self.name}'"
                )
            
            candidate = candidates[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            
            # Concatenate all text parts
            output_text = ''
            for part in parts:
                if 'text' in part:
                    output_text += part['text']
            
            # Build usage info if available
            usage = None
            if 'usageMetadata' in data:
                metadata = data['usageMetadata']
                usage = {
                    'prompt_tokens': metadata.get('promptTokenCount', 0),
                    'completion_tokens': metadata.get('candidatesTokenCount', 0),
                    'total_tokens': metadata.get('totalTokenCount', 0),
                }
            
            return ProviderResponse(
                model=self.model,
                output_text=output_text,
                raw=data,
                usage=usage,
                finish_reason=candidate.get('finishReason'),
            )
        
        except Exception as e:
            if hasattr(e, 'response'):
                status_code = getattr(e.response, 'status_code', None)
                error_msg = f"Google API error (status {status_code}): {e}"
            else:
                error_msg = f"Google API error: {e}"
            
            raise ProviderError(error_msg) from e
    
    def supports_streaming(self) -> bool:
        """Google supports streaming."""
        return True
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """
        Stream generation tokens as they arrive.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
        
        Yields:
            Incremental text chunks
        
        Raises:
            ProviderError: If streaming fails
        """
        client = self._get_http_client()
        
        # Build URL for streaming
        if self._use_vertex:
            access_token = await self._get_access_token()
            headers = self._build_headers(access_token)
            url = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project}/locations/{self.location}/"
                f"publishers/google/models/{self.model}:streamGenerateContent"
            )
        else:
            headers = self._build_headers()
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:streamGenerateContent?key={self.api_key}"
        
        # Build payload
        converted_messages = self._convert_messages(messages)
        payload = {
            'contents': converted_messages,
            'generationConfig': {
                'temperature': kwargs.get('temperature', self.temperature),
                'maxOutputTokens': kwargs.get('max_tokens', self.max_tokens),
            },
        }
        
        try:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == '':
                        continue
                    
                    try:
                        import json
                        chunk_data = json.loads(line)
                        
                        candidates = chunk_data.get('candidates', [])
                        if candidates:
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            
                            for part in parts:
                                if 'text' in part:
                                    yield part['text']
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            raise ProviderError(f"Google streaming error: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Register this provider
register_provider_class('google', GoogleProvider)
register_provider_class('vertex', GoogleProvider)  # Alias


__all__ = ['GoogleProvider']
