"""Tests for LLM registry."""

import pytest
from namel3ss.llm.registry import LLMRegistry, get_registry, reset_registry
from namel3ss.llm.base import BaseLLM, LLMResponse, LLMError


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def generate(self, prompt, **kwargs):
        return LLMResponse(text='response', raw={}, model=self.model)
    
    def generate_chat(self, messages, **kwargs):
        return LLMResponse(text='chat response', raw={}, model=self.model)
    
    def supports_streaming(self):
        return False


def test_registry_creation():
    """Test registry creation."""
    registry = LLMRegistry()
    assert len(registry) == 0
    assert registry.list() == []


def test_registry_register():
    """Test registering an LLM."""
    registry = LLMRegistry()
    llm = MockLLM('test_llm', 'test-model')
    
    registry.register(llm)
    
    assert len(registry) == 1
    assert 'test_llm' in registry
    assert registry.list() == ['test_llm']


def test_registry_register_duplicate():
    """Test registering duplicate raises error."""
    registry = LLMRegistry()
    llm1 = MockLLM('test_llm', 'model-1')
    llm2 = MockLLM('test_llm', 'model-2')
    
    registry.register(llm1)
    
    with pytest.raises(ValueError, match='already registered'):
        registry.register(llm2)


def test_registry_update():
    """Test updating an LLM."""
    registry = LLMRegistry()
    llm1 = MockLLM('test_llm', 'model-1')
    llm2 = MockLLM('test_llm', 'model-2')
    
    registry.update(llm1)
    assert registry.get('test_llm').model == 'model-1'
    
    registry.update(llm2)
    assert registry.get('test_llm').model == 'model-2'


def test_registry_get():
    """Test retrieving an LLM."""
    registry = LLMRegistry()
    llm = MockLLM('test_llm', 'test-model')
    registry.register(llm)
    
    retrieved = registry.get('test_llm')
    assert retrieved is llm
    assert retrieved.name == 'test_llm'


def test_registry_get_missing():
    """Test retrieving missing LLM returns None."""
    registry = LLMRegistry()
    assert registry.get('missing') is None


def test_registry_get_required():
    """Test get_required returns LLM."""
    registry = LLMRegistry()
    llm = MockLLM('test_llm', 'test-model')
    registry.register(llm)
    
    retrieved = registry.get_required('test_llm')
    assert retrieved is llm


def test_registry_get_required_missing():
    """Test get_required raises error for missing LLM."""
    registry = LLMRegistry()
    
    with pytest.raises(LLMError, match='is not registered'):
        registry.get_required('missing')


def test_registry_has():
    """Test checking if LLM is registered."""
    registry = LLMRegistry()
    llm = MockLLM('test_llm', 'test-model')
    
    assert not registry.has('test_llm')
    
    registry.register(llm)
    
    assert registry.has('test_llm')


def test_registry_list():
    """Test listing registered LLMs."""
    registry = LLMRegistry()
    
    llm1 = MockLLM('llm_1', 'model-1')
    llm2 = MockLLM('llm_2', 'model-2')
    llm3 = MockLLM('llm_3', 'model-3')
    
    registry.register(llm1)
    registry.register(llm2)
    registry.register(llm3)
    
    names = registry.list()
    assert len(names) == 3
    assert 'llm_1' in names
    assert 'llm_2' in names
    assert 'llm_3' in names


def test_registry_clear():
    """Test clearing the registry."""
    registry = LLMRegistry()
    
    llm1 = MockLLM('llm_1', 'model-1')
    llm2 = MockLLM('llm_2', 'model-2')
    
    registry.register(llm1)
    registry.register(llm2)
    
    assert len(registry) == 2
    
    registry.clear()
    
    assert len(registry) == 0
    assert registry.list() == []


def test_registry_contains():
    """Test __contains__ operator."""
    registry = LLMRegistry()
    llm = MockLLM('test_llm', 'test-model')
    
    assert 'test_llm' not in registry
    
    registry.register(llm)
    
    assert 'test_llm' in registry


def test_get_global_registry():
    """Test getting global registry."""
    reset_registry()
    
    registry1 = get_registry()
    registry2 = get_registry()
    
    # Should return the same instance
    assert registry1 is registry2


def test_global_registry_persistence():
    """Test global registry persists across calls."""
    reset_registry()
    
    registry1 = get_registry()
    llm = MockLLM('test_llm', 'test-model')
    registry1.register(llm)
    
    registry2 = get_registry()
    assert registry2.has('test_llm')
    assert registry2.get('test_llm') is llm


def test_reset_registry():
    """Test resetting the global registry."""
    reset_registry()  # Start fresh
    
    registry = get_registry()
    llm = MockLLM('test_llm_unique_456', 'test-model')
    registry.register(llm)
    
    reset_registry()
    
    new_registry = get_registry()
    assert len(new_registry) == 0
