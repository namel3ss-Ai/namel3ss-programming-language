"""
Mock LLM providers for deterministic testing of namel3ss applications.

This module provides mock implementations of LLM providers that return
deterministic, configurable responses instead of making real API calls.
This enables reliable, offline testing of AI applications.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Union
from unittest.mock import MagicMock

from namel3ss.llm.base import BaseLLM, ChatMessage, LLMResponse
from namel3ss.providers.base import N3Provider, ProviderMessage, ProviderResponse
from namel3ss.testing import MockLLMSpec, MockLLMResponse


class MockLLMProvider(BaseLLM):
    """
    Mock LLM provider that returns deterministic responses for testing.
    
    Can be configured with response mappings based on:
    - Model name
    - Prompt content patterns (regex)
    - Message history
    
    Example usage:
        >>> mock_llm = MockLLMProvider()
        >>> mock_llm.add_response_mapping(
        ...     model="gpt-4",
        ...     prompt_pattern=r"analyze.*content",
        ...     response="This content appears positive with topics: work, feedback"
        ... )
        >>> response = mock_llm.generate("Analyze this content: Great work!")
        >>> response.output_text
        'This content appears positive with topics: work, feedback'
    """
    
    def __init__(self, name: str = "mock_llm"):
        """
        Initialize mock LLM provider.
        
        Args:
            name: Provider name for identification
        """
        super().__init__()
        self.name = name
        self.response_mappings: List[ResponseMapping] = []
        self.default_response = "Mock response"
        self.call_history: List[Dict[str, Any]] = []
        
    def add_response_mapping(
        self,
        model: Optional[str] = None,
        prompt_pattern: Optional[str] = None,
        message_pattern: Optional[List[Dict[str, str]]] = None,
        response: Union[str, MockLLMResponse] = "Mock response",
        priority: int = 0
    ) -> None:
        """
        Add a response mapping for specific conditions.
        
        Args:
            model: Model name to match (None matches any)
            prompt_pattern: Regex pattern to match prompt text (None matches any)
            message_pattern: Pattern to match message history (None matches any)
            response: Response to return when conditions match
            priority: Priority for matching (higher = checked first)
        """
        if isinstance(response, str):
            response = MockLLMResponse(output_text=response)
            
        mapping = ResponseMapping(
            model=model,
            prompt_pattern=re.compile(prompt_pattern) if prompt_pattern else None,
            message_pattern=message_pattern,
            response=response,
            priority=priority
        )
        
        # Insert in priority order (highest first)
        inserted = False
        for i, existing in enumerate(self.response_mappings):
            if mapping.priority > existing.priority:
                self.response_mappings.insert(i, mapping)
                inserted = True
                break
        if not inserted:
            self.response_mappings.append(mapping)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response from prompt (synchronous).
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (model, temperature, etc.)
            
        Returns:
            LLMResponse with mock output
        """
        model = kwargs.get("model", "default")
        
        # Record call for inspection
        call_record = {
            "timestamp": time.time(),
            "prompt": prompt,
            "model": model,
            "kwargs": kwargs,
            "type": "generate"
        }
        self.call_history.append(call_record)
        
        # Find matching response
        mock_response = self._find_response(model, prompt, [])
        
        # Simulate delay if configured
        if mock_response.delay_ms > 0:
            time.sleep(mock_response.delay_ms / 1000.0)
        
        return LLMResponse(
            output_text=mock_response.output_text,
            metadata={"mock": True, **mock_response.metadata}
        )
    
    def generate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Generate chat response from message history.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with mock output
        """
        model = kwargs.get("model", "default")
        
        # Convert messages for matching
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        latest_prompt = messages[-1].content if messages else ""
        
        # Record call
        call_record = {
            "timestamp": time.time(),
            "messages": message_dicts,
            "model": model,
            "kwargs": kwargs,
            "type": "generate_chat"
        }
        self.call_history.append(call_record)
        
        # Find matching response
        mock_response = self._find_response(model, latest_prompt, message_dicts)
        
        # Simulate delay
        if mock_response.delay_ms > 0:
            time.sleep(mock_response.delay_ms / 1000.0)
        
        return LLMResponse(
            output_text=mock_response.output_text,
            metadata={"mock": True, **mock_response.metadata}
        )
    
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Async generate response from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with mock output
        """
        # Simulate async behavior but return sync result
        response = self.generate(prompt, **kwargs)
        
        # Simulate async delay if configured
        mock_response = self._find_response(kwargs.get("model", "default"), prompt, [])
        if mock_response.delay_ms > 0:
            await asyncio.sleep(mock_response.delay_ms / 1000.0)
        
        return response
    
    async def agenerate_chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """
        Async generate chat response from messages.
        
        Args:
            messages: List of chat messages  
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with mock output
        """
        response = self.generate_chat(messages, **kwargs)
        
        # Simulate async delay
        latest_prompt = messages[-1].content if messages else ""
        mock_response = self._find_response(kwargs.get("model", "default"), latest_prompt, [])
        if mock_response.delay_ms > 0:
            await asyncio.sleep(mock_response.delay_ms / 1000.0)
        
        return response
    
    def _find_response(
        self, 
        model: str, 
        prompt: str, 
        messages: List[Dict[str, str]]
    ) -> MockLLMResponse:
        """
        Find the best matching response for the given inputs.
        
        Args:
            model: Model name
            prompt: Prompt text
            messages: Message history
            
        Returns:
            MockLLMResponse to use
        """
        for mapping in self.response_mappings:
            if mapping.matches(model, prompt, messages):
                return mapping.response
        
        # Default response if no mapping matches
        return MockLLMResponse(output_text=self.default_response)
    
    def clear_history(self) -> None:
        """Clear call history for fresh test state."""
        self.call_history.clear()
    
    def get_call_count(self) -> int:
        """Get total number of calls made to this mock."""
        return len(self.call_history)
    
    def get_calls_for_model(self, model: str) -> List[Dict[str, Any]]:
        """Get all calls made for a specific model."""
        return [call for call in self.call_history if call.get("model") == model]


class MockN3Provider(N3Provider):
    """
    Mock N3Provider implementation for testing namel3ss runtime integration.
    
    Provides the same interface as real N3Provider but with deterministic
    responses configured for testing.
    """
    
    def __init__(self, name: str = "mock_n3_provider"):
        """
        Initialize mock N3Provider.
        
        Args:
            name: Provider name for identification
        """
        super().__init__()
        self.name = name
        self.response_mappings: List[ResponseMapping] = []
        self.default_response = ProviderResponse(output_text="Mock N3 response")
        self.call_history: List[Dict[str, Any]] = []
    
    def add_response_mapping(
        self,
        model: Optional[str] = None,
        prompt_pattern: Optional[str] = None,
        response: Union[str, ProviderResponse] = "Mock response",
        priority: int = 0
    ) -> None:
        """Add response mapping for provider calls."""
        if isinstance(response, str):
            response = ProviderResponse(output_text=response)
            
        mock_response = MockLLMResponse(output_text=response.output_text)
        
        mapping = ResponseMapping(
            model=model,
            prompt_pattern=re.compile(prompt_pattern) if prompt_pattern else None,
            message_pattern=None,
            response=mock_response,
            priority=priority
        )
        
        # Insert in priority order
        inserted = False
        for i, existing in enumerate(self.response_mappings):
            if mapping.priority > existing.priority:
                self.response_mappings.insert(i, mapping)
                inserted = True
                break
        if not inserted:
            self.response_mappings.append(mapping)
    
    async def generate(
        self, 
        messages: List[ProviderMessage], 
        **kwargs
    ) -> ProviderResponse:
        """
        Generate response using mock provider.
        
        Args:
            messages: List of provider messages
            **kwargs: Generation parameters
            
        Returns:
            ProviderResponse with mock output
        """
        model = kwargs.get("model", "default")
        latest_prompt = messages[-1].content if messages else ""
        
        # Record call
        call_record = {
            "timestamp": time.time(),
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "model": model,
            "kwargs": kwargs,
            "type": "generate"
        }
        self.call_history.append(call_record)
        
        # Find matching response
        mock_response = self._find_response(model, latest_prompt, [])
        
        # Simulate async delay
        if mock_response.delay_ms > 0:
            await asyncio.sleep(mock_response.delay_ms / 1000.0)
        
        return ProviderResponse(
            output_text=mock_response.output_text,
            raw={"mock": True, **mock_response.metadata}
        )
    
    def _find_response(
        self, 
        model: str, 
        prompt: str, 
        messages: List[Dict[str, str]]
    ) -> MockLLMResponse:
        """Find matching response for inputs."""
        for mapping in self.response_mappings:
            if mapping.matches(model, prompt, messages):
                return mapping.response
        
        return MockLLMResponse(output_text=self.default_response.output_text)
    
    def clear_history(self) -> None:
        """Clear call history."""
        self.call_history.clear()


@dataclass
class ResponseMapping:
    """
    Internal mapping from conditions to mock responses.
    
    Defines when a particular mock response should be used based on:
    - Model name matching
    - Prompt pattern matching
    - Message history patterns
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        prompt_pattern: Optional[Pattern[str]] = None,
        message_pattern: Optional[List[Dict[str, str]]] = None,
        response: MockLLMResponse = None,
        priority: int = 0
    ):
        self.model = model
        self.prompt_pattern = prompt_pattern
        self.message_pattern = message_pattern
        self.response = response or MockLLMResponse(output_text="Default mock")
        self.priority = priority
    
    def matches(
        self, 
        model: str, 
        prompt: str, 
        messages: List[Dict[str, str]]
    ) -> bool:
        """
        Check if this mapping matches the given inputs.
        
        Args:
            model: Model name to check
            prompt: Prompt text to check
            messages: Message history to check
            
        Returns:
            True if all conditions match
        """
        # Check model match
        if self.model is not None and self.model != model:
            return False
        
        # Check prompt pattern match
        if self.prompt_pattern is not None:
            if not self.prompt_pattern.search(prompt):
                return False
        
        # Check message pattern match (simplified for now)
        if self.message_pattern is not None:
            # Could implement more sophisticated message history matching
            # For now, just check if the pattern is a subset
            pass
        
        return True


def create_mock_llm_from_spec(spec: MockLLMSpec) -> MockLLMProvider:
    """
    Create a MockLLMProvider from a MockLLMSpec configuration.
    
    Args:
        spec: Mock LLM specification
        
    Returns:
        Configured MockLLMProvider
        
    Example:
        >>> spec = MockLLMSpec(
        ...     model_name="gpt-4",
        ...     prompt_pattern=r"analyze.*",
        ...     response=MockLLMResponse(output_text="Analysis complete")
        ... )
        >>> mock_llm = create_mock_llm_from_spec(spec)
        >>> response = mock_llm.generate("analyze this text", model="gpt-4")
        >>> response.output_text
        'Analysis complete'
    """
    mock_llm = MockLLMProvider()
    
    if spec.response:
        mock_llm.add_response_mapping(
            model=spec.model_name,
            prompt_pattern=spec.prompt_pattern,
            response=spec.response
        )
    elif spec.responses:
        # For multi-turn scenarios, add multiple responses
        for i, response in enumerate(spec.responses):
            # Use different patterns or priorities for sequencing
            mock_llm.add_response_mapping(
                model=spec.model_name,
                prompt_pattern=spec.prompt_pattern,
                response=response,
                priority=len(spec.responses) - i  # First response has highest priority
            )
    
    return mock_llm


__all__ = [
    "MockLLMProvider",
    "MockN3Provider", 
    "ResponseMapping",
    "create_mock_llm_from_spec"
]