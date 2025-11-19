"""Tests for base LLM interface and types."""

import pytest
from namel3ss.llm.base import BaseLLM, LLMResponse, LLMError, ChatMessage


def test_chat_message_creation():
    """Test ChatMessage creation and conversion."""
    msg = ChatMessage(role='user', content='Hello!')
    assert msg.role == 'user'
    assert msg.content == 'Hello!'
    assert msg.name is None
    assert msg.function_call is None
    
    msg_dict = msg.to_dict()
    assert msg_dict == {'role': 'user', 'content': 'Hello!'}


def test_chat_message_with_name():
    """Test ChatMessage with name."""
    msg = ChatMessage(role='assistant', content='Hi!', name='bot')
    msg_dict = msg.to_dict()
    assert msg_dict == {'role': 'assistant', 'content': 'Hi!', 'name': 'bot'}


def test_llm_response_creation():
    """Test LLMResponse creation."""
    response = LLMResponse(
        text='Paris',
        raw={'data': 'test'},
        model='gpt-4',
        usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        finish_reason='stop',
        metadata={'provider': 'openai'}
    )
    
    assert response.text == 'Paris'
    assert response.model == 'gpt-4'
    assert response.finish_reason == 'stop'
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15


def test_llm_response_no_usage():
    """Test LLMResponse without usage info."""
    response = LLMResponse(
        text='Hello',
        raw={},
        model='test-model'
    )
    
    # Properties return 0 when usage is None
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.total_tokens == 0


def test_llm_error():
    """Test LLMError exception."""
    error = LLMError(
        'API failed',
        provider='openai',
        model='gpt-4',
        status_code=500
    )
    
    assert str(error) == 'API failed'
    assert error.provider == 'openai'
    assert error.model == 'gpt-4'
    assert error.status_code == 500


def test_llm_error_with_original():
    """Test LLMError with original exception."""
    original = ValueError('Invalid token')
    error = LLMError('Request failed', original_error=original)
    
    assert error.original_error is original


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def generate(self, prompt, **kwargs):
        return LLMResponse(
            text=f'Response to: {prompt}',
            raw={},
            model=self.model
        )
    
    def generate_chat(self, messages, **kwargs):
        last_msg = messages[-1].content if messages else ''
        return LLMResponse(
            text=f'Chat response to: {last_msg}',
            raw={},
            model=self.model
        )
    
    def supports_streaming(self):
        return False


def test_base_llm_creation():
    """Test BaseLLM creation."""
    llm = MockLLM('test_llm', 'test-model')
    
    assert llm.name == 'test_llm'
    assert llm.model == 'test-model'
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1024
    assert llm.timeout == 60.0
    assert llm.get_provider_name() == 'mock'


def test_base_llm_with_config():
    """Test BaseLLM with custom config."""
    llm = MockLLM('test_llm', 'test-model', {
        'temperature': 0.5,
        'max_tokens': 2048,
        'timeout': 120.0,
        'custom_param': 'value'
    })
    
    assert llm.temperature == 0.5
    assert llm.max_tokens == 2048
    assert llm.timeout == 120.0
    assert llm.config['custom_param'] == 'value'


def test_base_llm_generate():
    """Test generate method."""
    llm = MockLLM('test_llm', 'test-model')
    response = llm.generate('What is 2+2?')
    
    assert response.text == 'Response to: What is 2+2?'
    assert response.model == 'test-model'


def test_base_llm_generate_chat():
    """Test generate_chat method."""
    llm = MockLLM('test_llm', 'test-model')
    messages = [
        ChatMessage(role='user', content='Hello'),
        ChatMessage(role='assistant', content='Hi!'),
        ChatMessage(role='user', content='How are you?'),
    ]
    response = llm.generate_chat(messages)
    
    assert response.text == 'Chat response to: How are you?'


def test_base_llm_generate_batch():
    """Test generate_batch default implementation."""
    llm = MockLLM('test_llm', 'test-model')
    prompts = ['Prompt 1', 'Prompt 2', 'Prompt 3']
    responses = llm.generate_batch(prompts)
    
    assert len(responses) == 3
    assert responses[0].text == 'Response to: Prompt 1'
    assert responses[1].text == 'Response to: Prompt 2'
    assert responses[2].text == 'Response to: Prompt 3'


def test_base_llm_streaming_not_supported():
    """Test that streaming raises error when not supported."""
    llm = MockLLM('test_llm', 'test-model')
    
    assert not llm.supports_streaming()
    
    with pytest.raises(NotImplementedError, match='does not support streaming'):
        list(llm.stream('test'))
    
    with pytest.raises(NotImplementedError, match='does not support streaming'):
        list(llm.stream_chat([ChatMessage(role='user', content='test')]))


def test_base_llm_repr():
    """Test __repr__ method."""
    llm = MockLLM('my_llm', 'gpt-4')
    assert repr(llm) == "MockLLM(name='my_llm', model='gpt-4')"
