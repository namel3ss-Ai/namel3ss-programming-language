"""
AI Provider System for N3 Language Development Assistance

Supports multiple AI backends with unified interface for:
- Code generation and completion
- Intelligent refactoring suggestions  
- Documentation generation
- Testing assistance
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Request for AI-powered code generation"""
    prompt: str
    context: Dict[str, Any] = None
    language: str = "n3"
    max_tokens: int = 1000
    temperature: float = 0.1
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.model_params is None:
            self.model_params = {}

@dataclass 
class GenerationResponse:
    """Response from AI code generation"""
    generated_code: str
    confidence: float
    reasoning: str = ""
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.config = kwargs
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def generate_code(self, request: GenerationRequest) -> GenerationResponse:
        """Generate code based on request"""
        pass
    
    @abstractmethod
    async def complete_code(self, context: str, cursor_position: int) -> List[str]:
        """Provide code completions"""
        pass
    
    @abstractmethod
    async def explain_code(self, code: str) -> str:
        """Explain what code does"""
        pass
    
    @abstractmethod
    async def suggest_refactoring(self, code: str) -> List[str]:
        """Suggest code improvements"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class OpenAIProvider(AIProvider):
    """OpenAI GPT provider for code assistance"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key or os.getenv("OPENAI_API_KEY"), **kwargs)
        
    def _setup(self) -> None:
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.available = True
        except ImportError:
            logger.warning("OpenAI library not available. Install with: pip install openai")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to setup OpenAI: {e}")
            self.available = False
    
    async def generate_code(self, request: GenerationRequest) -> GenerationResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
            
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                **request.model_params
            )
            
            content = response.choices[0].message.content.strip()
            return GenerationResponse(
                generated_code=self._extract_code(content),
                confidence=0.8,
                reasoning=self._extract_reasoning(content),
                metadata={"model": self.model, "tokens": response.usage.total_tokens}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def complete_code(self, context: str, cursor_position: int) -> List[str]:
        if not self.is_available():
            return []
            
        before_cursor = context[:cursor_position]
        after_cursor = context[cursor_position:]
        
        prompt = f"""Complete the N3 code at the cursor position:

BEFORE CURSOR:
{before_cursor}

AFTER CURSOR:
{after_cursor}

Provide 3 intelligent completions that make sense in this context.
Focus on N3 syntax and semantic understanding.
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
                n=3
            )
            
            return [choice.message.content.strip() for choice in response.choices]
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            return []
    
    async def explain_code(self, code: str) -> str:
        if not self.is_available():
            return "AI explanation not available"
            
        prompt = f"""Explain this N3 code in clear, simple terms:

```n3
{code}
```

Focus on:
- What the code does
- Key N3 language features used
- Any notable patterns or design choices
- How it fits into a larger application
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI explanation failed: {e}")
            return "Error generating explanation"
    
    async def suggest_refactoring(self, code: str) -> List[str]:
        if not self.is_available():
            return []
            
        prompt = f"""Analyze this N3 code and suggest 3-5 specific refactoring improvements:

```n3
{code}
```

Consider:
- Code organization and structure
- Performance optimizations
- N3 best practices and idioms
- Maintainability improvements
- Potential bug fixes

Provide concrete, actionable suggestions.
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2
            )
            
            suggestions = response.choices[0].message.content.strip()
            return [s.strip() for s in suggestions.split('\n') if s.strip() and s.strip().startswith('-')]
            
        except Exception as e:
            logger.error(f"OpenAI refactoring failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return self.available and self.api_key is not None
    
    def _build_system_prompt(self, request: GenerationRequest) -> str:
        return """You are an expert N3 programming language assistant. N3 is a modern, declarative language for building applications with:

- Pages: UI components with routing (page Home at "/" { ... })
- Frames: Data structures (frame User { id: int, name: string })  
- Apps: Application definitions (app MyApp { ... })
- Components: Reusable UI elements
- Styling: Built-in CSS-like styling
- State management: Reactive state handling

Generate clean, idiomatic N3 code that follows best practices. Always include comments explaining key concepts."""
    
    def _build_user_prompt(self, request: GenerationRequest) -> str:
        context_str = ""
        if request.context:
            context_str = f"\nContext:\n{request.context}\n"
            
        return f"{context_str}Task: {request.prompt}\n\nGenerate N3 code that accomplishes this task."
    
    def _extract_code(self, content: str) -> str:
        """Extract code from response content"""
        lines = content.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # If no code block, return content as-is
        return content
    
    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning/explanation from response"""
        lines = content.split('\n')
        reasoning_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip():
                reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines)

class AnthropicProvider(AIProvider):
    """Anthropic Claude provider for code assistance"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key or os.getenv("ANTHROPIC_API_KEY"), **kwargs)
    
    def _setup(self) -> None:
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.available = True
        except ImportError:
            logger.warning("Anthropic library not available. Install with: pip install anthropic")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to setup Anthropic: {e}")
            self.available = False
    
    async def generate_code(self, request: GenerationRequest) -> GenerationResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")
            
        prompt = self._build_claude_prompt(request)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": prompt}],
                **request.model_params
            )
            
            content = response.content[0].text.strip()
            return GenerationResponse(
                generated_code=self._extract_code(content),
                confidence=0.85,
                reasoning=self._extract_reasoning(content),
                metadata={"model": self.model, "tokens": response.usage.input_tokens + response.usage.output_tokens}
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def complete_code(self, context: str, cursor_position: int) -> List[str]:
        # Similar implementation to OpenAI but using Anthropic API
        if not self.is_available():
            return []
        
        # Implementation similar to OpenAI but with Anthropic-specific API calls
        before_cursor = context[:cursor_position]
        after_cursor = context[cursor_position:]
        
        prompt = f"""Complete the N3 code at the cursor position:

BEFORE CURSOR:
{before_cursor}

AFTER CURSOR:  
{after_cursor}

Provide intelligent completions for N3 syntax."""
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return [response.content[0].text.strip()]
            
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            return []
    
    async def explain_code(self, code: str) -> str:
        # Similar to OpenAI implementation
        if not self.is_available():
            return "AI explanation not available"
        
        prompt = f"Explain this N3 code clearly:\n\n```n3\n{code}\n```"
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic explanation failed: {e}")
            return "Error generating explanation"
    
    async def suggest_refactoring(self, code: str) -> List[str]:
        # Similar to OpenAI implementation
        if not self.is_available():
            return []
        
        prompt = f"Suggest refactoring improvements for this N3 code:\n\n```n3\n{code}\n```"
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            suggestions = response.content[0].text.strip()
            return [s.strip() for s in suggestions.split('\n') if s.strip() and s.strip().startswith('-')]
            
        except Exception as e:
            logger.error(f"Anthropic refactoring failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return self.available and self.api_key is not None
    
    def _build_claude_prompt(self, request: GenerationRequest) -> str:
        context_str = ""
        if request.context:
            context_str = f"Context: {request.context}\n\n"
            
        return f"""{context_str}You are an expert N3 programming language assistant. Generate clean, idiomatic N3 code.

Task: {request.prompt}

N3 Language Features:
- Pages: page Home at "/" {{ content here }}
- Frames: frame User {{ id: int, name: string }}
- Apps: app MyApp {{ pages, styling }}
- Components: Reusable UI elements  
- Styling: Built-in CSS-like styling

Generate N3 code with explanations."""
    
    def _extract_code(self, content: str) -> str:
        """Extract code from Claude response"""
        lines = content.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return content
    
    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from Claude response"""
        lines = content.split('\n')
        reasoning_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip():
                reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines)

class OllamaProvider(AIProvider):
    """Ollama local AI provider for code assistance"""
    
    def __init__(self, model: str = "codellama:7b", host: str = "http://localhost:11434", **kwargs):
        self.host = host
        super().__init__(model, api_key=None, **kwargs)
    
    def _setup(self) -> None:
        try:
            import httpx
            self.client = httpx.AsyncClient()
            self.available = True
        except ImportError:
            logger.warning("httpx library not available. Install with: pip install httpx")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to setup Ollama: {e}")
            self.available = False
    
    async def generate_code(self, request: GenerationRequest) -> GenerationResponse:
        if not self.is_available():
            raise RuntimeError("Ollama provider not available")
            
        prompt = self._build_ollama_prompt(request)
        
        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "").strip()
                
                return GenerationResponse(
                    generated_code=self._extract_code(content),
                    confidence=0.7,
                    reasoning=self._extract_reasoning(content),
                    metadata={"model": self.model, "host": self.host}
                )
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def complete_code(self, context: str, cursor_position: int) -> List[str]:
        if not self.is_available():
            return []
        
        before_cursor = context[:cursor_position]
        prompt = f"Complete this N3 code:\n{before_cursor}"
        
        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 100}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                completion = result.get("response", "").strip()
                return [completion] if completion else []
            
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
        
        return []
    
    async def explain_code(self, code: str) -> str:
        if not self.is_available():
            return "AI explanation not available"
        
        prompt = f"Explain this N3 code:\n\n{code}\n\nExplanation:"
        
        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 300}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama explanation failed: {e}")
        
        return "Error generating explanation"
    
    async def suggest_refactoring(self, code: str) -> List[str]:
        if not self.is_available():
            return []
        
        prompt = f"Suggest improvements for this N3 code:\n\n{code}\n\nSuggestions:"
        
        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                suggestions = result.get("response", "").strip()
                return [s.strip() for s in suggestions.split('\n') if s.strip() and s.strip().startswith('-')]
            
        except Exception as e:
            logger.error(f"Ollama refactoring failed: {e}")
        
        return []
    
    async def is_available(self) -> bool:
        if not self.available:
            return False
        
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def _build_ollama_prompt(self, request: GenerationRequest) -> str:
        context_str = ""
        if request.context:
            context_str = f"Context: {request.context}\n\n"
            
        return f"""{context_str}Task: Generate N3 code for: {request.prompt}

N3 Language Features:
- Pages: page Home at "/" {{ content }}  
- Frames: frame User {{ id: int, name: string }}
- Apps: app MyApp {{ pages and config }}

Generate clean N3 code:"""
    
    def _extract_code(self, content: str) -> str:
        """Extract code from Ollama response"""
        lines = content.split('\n')
        code_lines = []
        
        # Look for code blocks or assume content is code
        for line in lines:
            if line.strip() and not line.startswith('#') and not line.startswith('//'):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from Ollama response"""
        lines = content.split('\n')
        reasoning_lines = []
        
        for line in lines:
            if line.startswith('#') or line.startswith('//') or 'explanation' in line.lower():
                reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines)

# Provider factory
def create_provider(provider_type: str, **kwargs) -> AIProvider:
    """Create an AI provider instance"""
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(providers.keys())}")
    
    return providers[provider_type](**kwargs)

# Auto-detect available providers
async def detect_available_providers() -> List[str]:
    """Detect which AI providers are available"""
    available = []
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            provider = OpenAIProvider()
            if provider.is_available():
                available.append("openai")
        except:
            pass
    
    # Check Anthropic  
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            provider = AnthropicProvider()
            if provider.is_available():
                available.append("anthropic")
        except:
            pass
    
    # Check Ollama
    try:
        provider = OllamaProvider()
        if await provider.is_available():
            available.append("ollama")
    except:
        pass
    
    return available