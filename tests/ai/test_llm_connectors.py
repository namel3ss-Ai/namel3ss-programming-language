"""Tests for provider-specific LLM connector behaviour with mocked HTTP flows."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Dict

import pytest

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser

from tests.backend_test_utils import load_backend_module


@pytest.fixture
def runtime_module(tmp_path, monkeypatch):
    source = (
        'app "LLMTest".\n'
        '\n'
        'connector "openai" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o-mini"\n'
        '\n'
        'connector "anthropic" type llm:\n'
        '  provider = "anthropic"\n'
        '  model = "claude-3-sonnet"\n'
        '\n'
        'connector "azure" type llm:\n'
        '  provider = "openai"\n'
        '  model = "gpt-4o"\n'
        '  deployment = "gpt-prod"\n'
        '  api_base = "https://example.azure.com/openai"\n'
        '\n'
        'model "chat_model" using openai:\n'
        '  name: "gpt-4o-mini"\n'
        '\n'
        'prompt "SummarizeTicket":\n'
        '  input:\n'
        '    ticket: text\n'
        '  output:\n'
        '    summary: text\n'
        '  using model "chat_model":\n'
        '    "Summary: {{ticket}}"\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Hello"\n'
    )
    app = Parser(source).parse_app()
    backend_dir = tmp_path / "backend_llm"
    generate_backend(app, backend_dir)

    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        yield module


@pytest.fixture(autouse=True)
def _patch_ai_connectors(runtime_module):
    runtime = runtime_module.runtime
    connectors = {
        "openai": {
            "config": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
        "anthropic": {
            "config": {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "api_key_env": "ANTHROPIC_API_KEY",
                "system": "You are testing",
            }
        },
        "azure": {
            "config": {
                "provider": "openai",
                "model": "gpt-4o",
                "deployment": "gpt-prod",
                "api_key_env": "AZURE_OPENAI_KEY",
                "api_base": "https://example.azure.com/openai",
                "api_version": "2024-05-01-preview",
            }
        },
    }
    runtime.AI_CONNECTORS = connectors
    runtime_module.AI_CONNECTORS = connectors
    env_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_KEY",
    ]
    runtime.ENV_KEYS = env_keys
    runtime_module.ENV_KEYS = env_keys
    return connectors


@pytest.fixture(autouse=True)
def _mock_http_post(monkeypatch, runtime_module):
    def _fake_post(url: str, data: Dict[str, object], headers: Dict[str, str], timeout: float):
        if "anthropic.com" in url:
            payload = {
                "content": [
                    {
                        "type": "text",
                        "text": "Anthropic mock response",
                    }
                ]
            }
        elif "deployments" in url:
            payload = {
                "choices": [
                    {
                        "text": "Azure deployment text",
                    }
                ]
            }
        else:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from test",
                        }
                    }
                ],
            }
        raw_text = json.dumps(payload)
        return 200, raw_text, payload

    if hasattr(runtime_module, "_http_post_json"):
        monkeypatch.setattr(runtime_module, "_http_post_json", _fake_post)
    monkeypatch.setattr(runtime_module.runtime, "_http_post_json", _fake_post)
    return _fake_post


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-azure")


def test_openai_connector_returns_structured_text(runtime_module):
    result = runtime_module.call_llm_connector("openai", {"prompt": "Greetings"})
    assert result["status"] == "ok"
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o-mini"
    assert result["result"]["text"] == "Hello from test"
    assert result["result"]["json"]["choices"][0]["message"]["content"] == "Hello from test"
    assert result["metadata"]["http_status"] == 200


def test_anthropic_connector_formats_messages(runtime_module):
    result = runtime_module.call_llm_connector("anthropic", {"prompt": "Explain"})
    assert result["status"] == "ok"
    assert result["provider"] == "anthropic"
    assert "Anthropic mock response" in result["result"]["text"]
    assert result["metadata"]["http_status"] == 200


def test_azure_deployment_connector_builds_endpoint(runtime_module):
    result = runtime_module.call_llm_connector("azure", {"prompt": "Azure"})
    assert result["status"] == "ok"
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"
    assert result["result"]["text"] == "Azure deployment text"
    assert "raw" in result["result"]


def test_call_llm_connector_missing_api_key_errors(runtime_module, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    result = runtime_module.call_llm_connector("openai", {"prompt": "Hi"})

    assert result["status"] == "error"
    assert "OpenAI API key is missing" in result["error"]
    assert "metadata" in result
    assert "result" not in result


def test_call_llm_connector_missing_api_key_stub_when_enabled(runtime_module, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "1")

    result = runtime_module.call_llm_connector("openai", {"prompt": "Hi"})

    assert result["status"] == "stub"
    assert result["result"]["text"] == "[stub: llm call failed]"
    assert "config" in result and result["config"]["provider"] == "openai"


def test_call_llm_connector_http_error_reports_status(runtime_module, monkeypatch):
    def _raise_http(url, data, headers, timeout):
        fp = io.BytesIO(b"service down")
        fp.seek(0)
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=fp)

    if hasattr(runtime_module, "_http_post_json"):
        monkeypatch.setattr(runtime_module, "_http_post_json", _raise_http)
    monkeypatch.setattr(runtime_module.runtime, "_http_post_json", _raise_http)
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    result = runtime_module.call_llm_connector("openai", {"prompt": "Ping"})

    assert result["status"] == "error"
    assert result["metadata"]["http_status"] == 503


def test_run_prompt_returns_structured_output(runtime_module, monkeypatch):
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    result = runtime_module.run_prompt("SummarizeTicket", {"ticket": "Help me"})

    assert result["status"] == "ok"
    assert result["prompt"] == "SummarizeTicket"
    assert result["output"]["summary"] == "Hello from test"
    assert result["metadata"]["http_status"] == 200


def test_run_prompt_missing_input_errors(runtime_module, monkeypatch):
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    result = runtime_module.run_prompt("SummarizeTicket", {})

    assert result["status"] == "error"
    assert "missing required prompt field" in result["error"].lower()


def test_call_llm_connector_http_error_stub_when_enabled(runtime_module, monkeypatch):
    def _raise_http(url, data, headers, timeout):
        fp = io.BytesIO(b"bad")
        fp.seek(0)
        raise urllib.error.HTTPError(url, 500, "Internal", hdrs=None, fp=fp)

    if hasattr(runtime_module, "_http_post_json"):
        monkeypatch.setattr(runtime_module, "_http_post_json", _raise_http)
    monkeypatch.setattr(runtime_module.runtime, "_http_post_json", _raise_http)
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "true")

    result = runtime_module.call_llm_connector("openai", {"prompt": "Ping"})

    assert result["status"] == "stub"
    assert result["result"]["text"] == "[stub: llm call failed]"
    assert result["metadata"]["http_status"] == 500


def test_run_chain_missing_reports_not_found(runtime_module, monkeypatch):
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    result = runtime_module.run_chain("missing_chain", {"input": "hello"})

    assert result["status"] == "not_found"
    assert result["result"] is None
    assert result["error"].startswith("Chain 'missing_chain'")


def test_run_chain_missing_without_error_when_stubs_enabled(runtime_module, monkeypatch):
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "1")

    result = runtime_module.run_chain("missing_chain", {"input": "hello"})

    assert result["status"] == "not_found"
    assert result["result"] is None
    assert "error" not in result
