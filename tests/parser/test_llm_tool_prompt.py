"""Tests for LLM, Tool, and Prompt parsing."""

import pytest

from namel3ss.parser import Parser


def test_parse_llm_block_openai():
    """Test parsing an OpenAI LLM definition."""
    source = '''
app "TestApp".

llm chat_gpt:
    provider: openai
    model: gpt-4o
    temperature: 0.7
    max_tokens: 2048
'''
    module = Parser(source).parse()
    assert len(module.body) == 1
    app = module.body[0]
    
    assert len(app.llms) == 1
    llm = app.llms[0]
    assert llm.name == "chat_gpt"
    assert llm.provider == "openai"
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2048


def test_parse_llm_block_anthropic_with_optional_params():
    """Test parsing Anthropic LLM with optional parameters."""
    source = '''
app "TestApp".

llm claude:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.8
    max_tokens: 4096
    top_p: 0.9
    frequency_penalty: 0.5
'''
    module = Parser(source).parse()
    app = module.body[0]
    
    assert len(app.llms) == 1
    llm = app.llms[0]
    assert llm.name == "claude"
    assert llm.provider == "anthropic"
    assert llm.model == "claude-3-5-sonnet-20241022"
    assert llm.temperature == 0.8
    assert llm.max_tokens == 4096
    assert llm.top_p == 0.9
    assert llm.frequency_penalty == 0.5


def test_parse_tool_block_http():
    """Test parsing an HTTP tool definition."""
    source = '''
app "TestApp".

tool get_weather:
    type: http
    endpoint: https://api.weather.com/v1/current
    method: GET
    timeout: 10.0
'''
    module = Parser(source).parse()
    app = module.body[0]
    
    assert len(app.tools) == 1
    tool = app.tools[0]
    assert tool.name == "get_weather"
    assert tool.type == "http"
    assert tool.endpoint == "https://api.weather.com/v1/current"
    assert tool.method == "GET"
    assert tool.timeout == 10.0


def test_parse_tool_block_with_headers():
    """Test parsing a tool with custom headers."""
    source = '''
app "TestApp".

tool search_api:
    type: http
    endpoint: https://api.search.com/v1/query
    method: POST
    timeout: 15.0
'''
    module = Parser(source).parse()
    app = module.body[0]
    
    assert len(app.tools) == 1
    tool = app.tools[0]
    assert tool.name == "search_api"
    assert tool.type == "http"
    assert tool.endpoint == "https://api.search.com/v1/query"
    assert tool.method == "POST"
    assert tool.timeout == 15.0


def test_parse_prompt_block_simple():
    """Test parsing a simple prompt definition."""
    source = '''
app "TestApp".

llm gpt4:
    provider: openai
    model: gpt-4o

prompt summarize:
    model: gpt4
    template: "Summarize the following text: {text}"
'''
    module = Parser(source).parse()
    app = module.body[0]
    
    assert len(app.prompts) == 1
    prompt = app.prompts[0]
    assert prompt.name == "summarize"
    assert prompt.model == "gpt4"
    assert prompt.template == "Summarize the following text: {text}"


def test_parse_multiple_llms_tools_prompts():
    """Test parsing multiple LLMs, tools, and prompts together."""
    source = '''
app "ComprehensiveApp".

llm gpt4:
    provider: openai
    model: gpt-4o
    temperature: 0.7

llm claude:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.8

tool weather:
    type: http
    endpoint: https://api.weather.com/v1
    method: GET

tool search:
    type: http
    endpoint: https://api.search.com/v1
    method: POST

prompt summarize:
    model: gpt4
    template: "Summarize: {text}"

prompt classify:
    model: claude
    template: "Classify: {text}"
'''
    module = Parser(source).parse()
    app = module.body[0]
    
    assert len(app.llms) == 2
    assert app.llms[0].name == "gpt4"
    assert app.llms[1].name == "claude"
    
    assert len(app.tools) == 2
    assert app.tools[0].name == "weather"
    assert app.tools[1].name == "search"
    
    assert len(app.prompts) == 2
    assert app.prompts[0].name == "summarize"
    assert app.prompts[1].name == "classify"


def test_llm_block_missing_provider_raises():
    """Test that missing required fields raise errors."""
    source = '''
app "TestApp".

llm incomplete:
    model: gpt-4o
'''
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    assert "provider" in str(exc_info.value).lower()


def test_llm_block_invalid_provider_raises():
    """Test that invalid provider raises error."""
    source = '''
app "TestApp".

llm invalid:
    provider: invalid_provider
    model: some-model
'''
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    assert "invalid provider" in str(exc_info.value).lower()
