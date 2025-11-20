"""Production-grade memory summarization for agent runtime."""

import asyncio
import logging
from typing import List, Optional, Protocol, Dict, Any
from dataclasses import dataclass

from namel3ss.ml.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from namel3ss.ml.providers.base import LLMError
from namel3ss.templates import get_default_engine, TemplateError


logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message structure for summarization (matches runtime.AgentMessage)."""
    
    role: str
    content: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


class BaseSummarizer(Protocol):
    """
    Protocol for conversation summarizers.
    
    Summarizers compress conversation history into concise summaries
    while preserving key information for context.
    """
    
    async def summarize(
        self,
        messages: List[AgentMessage],
        existing_summary: Optional[str] = None,
        *,
        max_tokens: int = 512,
    ) -> str:
        """
        Summarize a list of messages into a concise summary.
        
        Args:
            messages: List of messages to summarize
            existing_summary: Optional existing summary to build upon (incremental)
            max_tokens: Maximum tokens for the summary
        
        Returns:
            Concise summary string
        
        Raises:
            LLMError: If summarization fails
        """
        ...


class LLMSummarizer:
    """
    LLM-based summarizer using production-ready provider infrastructure.
    
    Supports incremental summarization by incorporating existing summaries
    and new conversation segments.
    """
    
    SYSTEM_PROMPT = """You are an expert at summarizing conversations while preserving critical information.

Your task:
1. Create a concise summary of the conversation that captures:
   - Key topics discussed
   - Important decisions or conclusions
   - Critical context needed for future turns
   - User goals and agent responses

2. If an existing summary is provided, incorporate new messages into it, avoiding redundancy.

3. Be extremely concise but do not lose important factual details.

4. Use third-person past tense (e.g., "The user asked about...", "The agent explained...").

5. Keep the summary under {max_tokens} tokens."""
    
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **config
    ):
        """
        Initialize LLM summarizer.
        
        Args:
            provider: Provider name ("openai", "anthropic", "ollama")
            model: Model name (e.g., "gpt-4o-mini", "claude-3-haiku-20240307", "llama3")
            api_key: API key (if required)
            base_url: Base URL (if required)
            **config: Additional provider configuration
        """
        self.provider = provider
        self.model = model
        self._llm = None
        self._api_key = api_key
        self._base_url = base_url
        self._config = config
    
    async def _get_llm(self):
        """Lazy initialization of LLM provider."""
        if self._llm is None:
            if self.provider == "openai":
                self._llm = OpenAIProvider(
                    model=self.model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    temperature=0.3,  # Lower temperature for consistent summaries
                    max_tokens=self._config.get("max_tokens", 1024),
                    **self._config
                )
            elif self.provider == "anthropic":
                self._llm = AnthropicProvider(
                    model=self.model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    temperature=0.3,
                    max_tokens=self._config.get("max_tokens", 1024),
                    **self._config
                )
            elif self.provider == "ollama":
                self._llm = OllamaProvider(
                    model=self.model,
                    base_url=self._base_url or "http://localhost:11434",
                    temperature=0.3,
                    max_tokens=self._config.get("max_tokens", 1024),
                    **self._config
                )
            else:
                raise LLMError(
                    f"Unsupported summarization provider: {self.provider}",
                    provider=self.provider
                )
        
        return self._llm
    
    def _format_messages_for_summary(self, messages: List[AgentMessage]) -> str:
        """Format messages into text for summarization."""
        lines = []
        
        for msg in messages:
            role = msg.role.upper()
            content = msg.content.strip()
            
            if msg.tool_call:
                tool_name = msg.tool_call.get("tool", "unknown")
                lines.append(f"[{role}] Called tool: {tool_name}")
            elif msg.tool_result:
                status = msg.tool_result.get("status", "unknown")
                lines.append(f"[{role}] Tool result: {status}")
            elif content:
                # Truncate very long messages for summarization prompt
                if len(content) > 500:
                    content = content[:497] + "..."
                lines.append(f"[{role}] {content}")
        
        return "\n".join(lines)
    
    def _build_summarization_prompt(
        self,
        messages: List[AgentMessage],
        existing_summary: Optional[str],
        max_tokens: int,
    ) -> str:
        """Build the prompt for summarization."""
        conversation_text = self._format_messages_for_summary(messages)
        
        if existing_summary:
            prompt = f"""Existing summary:
{existing_summary}

New conversation segment to incorporate:
{conversation_text}

Create an updated summary that incorporates the new information while maintaining the existing context. Keep it under {max_tokens} tokens."""
        else:
            prompt = f"""Conversation to summarize:
{conversation_text}

Create a concise summary under {max_tokens} tokens."""
        
        return prompt
    
    async def summarize(
        self,
        messages: List[AgentMessage],
        existing_summary: Optional[str] = None,
        *,
        max_tokens: int = 512,
    ) -> str:
        """
        Summarize messages using the configured LLM.
        
        Args:
            messages: List of messages to summarize
            existing_summary: Optional existing summary for incremental summarization
            max_tokens: Maximum tokens for the summary
        
        Returns:
            Concise summary string
        
        Raises:
            LLMError: If summarization fails
        """
        if not messages:
            return existing_summary or ""
        
        try:
            llm = await self._get_llm()
            
            # Build summarization prompt using template engine
            engine = get_default_engine()
            system_template = engine.compile(self.SYSTEM_PROMPT, name="summarization_system_prompt")
            system_prompt = system_template.render({"max_tokens": max_tokens})
            
            user_prompt = self._build_summarization_prompt(
                messages,
                existing_summary,
                max_tokens
            )
            
            # Generate summary
            logger.info(
                f"Summarizing {len(messages)} messages with {self.provider}/{self.model}"
            )
            
            response = await llm.agenerate(
                user_prompt,
                system=system_prompt,
                max_tokens=max_tokens
            )
            
            summary = response.content.strip()
            
            logger.info(
                f"Generated summary: {len(summary)} chars, "
                f"{response.usage.get('total_tokens', 0)} tokens"
            )
            
            return summary
        
        except Exception as e:
            logger.error(
                f"Summarization failed with {self.provider}/{self.model}: {e}",
                exc_info=True
            )
            raise LLMError(
                f"Summarization failed: {e}",
                provider=self.provider,
                model=self.model,
                original_error=e
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_llm()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._llm and hasattr(self._llm, '_close_client'):
            await self._llm._close_client()


def get_summarizer(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseSummarizer:
    """
    Factory function to create summarizers from configuration.
    
    Args:
        name: Logical name for summarizer (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku", "ollama/llama3")
        config: Optional configuration dictionary
    
    Returns:
        BaseSummarizer implementation
    
    Raises:
        ValueError: If name format is invalid
        LLMError: If provider is unsupported
    
    Examples:
        >>> summarizer = get_summarizer("openai/gpt-4o-mini")
        >>> summarizer = get_summarizer("anthropic/claude-3-haiku-20240307")
        >>> summarizer = get_summarizer("ollama/llama3", {"base_url": "http://localhost:11434"})
    """
    config = config or {}
    
    # Parse provider/model format
    if "/" not in name:
        raise ValueError(
            f"Invalid summarizer name format: '{name}'. "
            "Expected 'provider/model' (e.g., 'openai/gpt-4o-mini')"
        )
    
    provider, model = name.split("/", 1)
    provider = provider.lower()
    
    # Map common aliases
    provider_map = {
        "openai": "openai",
        "gpt": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "local": "ollama",
    }
    
    provider = provider_map.get(provider, provider)
    
    # Extract config
    api_key = config.get("api_key")
    base_url = config.get("base_url")
    extra_config = {k: v for k, v in config.items() if k not in ("api_key", "base_url")}
    
    return LLMSummarizer(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **extra_config
    )


__all__ = [
    "BaseSummarizer",
    "LLMSummarizer",
    "get_summarizer",
    "AgentMessage",
]
