from __future__ import annotations

import copy
import importlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

from namel3ss.codegen.backend.core.runtime_sections.prediction import PREDICTION_SECTION
from namel3ss.codegen.backend.core.runtime_sections.registry import REGISTRY_SECTION


def _make_prediction_namespace() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "copy": copy,
        "os": os,
        "Path": Path,
        "pickle": __import__("pickle"),
        "Dict": Dict,
        "Any": Any,
        "List": List,
        "logger": logging.getLogger("test_prediction"),
        "MODEL_REGISTRY": {},
        "MODELS": {},
        "MODEL_CACHE": {},
        "MODEL_LOADERS": {},
        "MODEL_RUNNERS": {},
        "MODEL_EXPLAINERS": {},
        "_load_python_callable": lambda _path: None,
        "REALTIME_ENABLED": False,
        "BROADCAST": SimpleNamespace(broadcast=lambda *args, **kwargs: None),
    }
    exec(PREDICTION_SECTION, namespace)  # pylint: disable=exec-used
    return namespace


def _make_registry_namespace() -> Dict[str, Any]:
    collections_abc = __import__("collections.abc", fromlist=("Awaitable",))

    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "os": os,
        "sys": sys,
        "Path": Path,
        "importlib": importlib,
        "logger": logging.getLogger("test_registry"),
        "time": __import__("time"),
        "Dict": Dict,
        "Any": Any,
        "Callable": Callable,
        "Awaitable": collections_abc.Awaitable,
        "List": List,
        "AsyncSession": SimpleNamespace,
    }
    exec(REGISTRY_SECTION, namespace)  # pylint: disable=exec-used
    return namespace


def _configure_model(namespace: Dict[str, Any], runner: Callable[..., Any]) -> str:
    model_name = "demo"
    namespace["MODEL_REGISTRY"][model_name] = {
        "type": "python",
        "framework": "generic",
        "version": "v1",
        "metadata": {},
    }
    namespace["MODEL_CACHE"][model_name] = SimpleNamespace(name=model_name)
    namespace["MODEL_RUNNERS"]["generic"] = runner
    namespace["MODEL_RUNNERS"]["python"] = runner
    return model_name


def test_predict_canonicalizes_runner_outputs() -> None:
    ns = _make_prediction_namespace()
    calls: List[Dict[str, Any]] = []

    def runner(model_name: str, instance: Any, payload: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(payload)
        return {"prob": 0.81, "label": "cat"}

    model_name = _configure_model(ns, runner)
    predict = ns["predict"]

    result = predict(model_name, {"feature": "value"})

    assert result["status"] == "ok"
    assert pytest.approx(result["output"]["score"], rel=1e-6) == 0.81
    assert result["output"]["label"] == "cat"
    assert "explanations" not in result
    assert result["metadata"]["runner"] == runner.__name__
    assert calls == [{"feature": "value"}]


def test_predict_marks_partial_without_fabrication() -> None:
    ns = _make_prediction_namespace()

    def runner(model_name: str, instance: Any, payload: Dict[str, Any], spec: Dict[str, Any]) -> str:
        return "unstructured"

    model_name = _configure_model(ns, runner)
    predict = ns["predict"]

    result = predict(model_name, {})

    assert result["status"] == "partial"
    assert result["output"]["status"] == "partial"
    assert result["output"]["score"] is None
    assert result["output"]["label"] == "unstructured"
    assert "explanations" not in result


def test_predict_generates_numeric_explanations() -> None:
    ns = _make_prediction_namespace()
    calls: List[Dict[str, Any]] = []

    def runner(model_name: str, instance: Any, payload: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(payload)
        return {"score": float(payload["x"])}

    model_name = _configure_model(ns, runner)
    predict = ns["predict"]

    result = predict(model_name, {"x": 2.0})

    assert result["status"] == "ok"
    assert result["output"]["score"] == pytest.approx(2.0)
    explanations = result.get("explanations")
    assert explanations is not None
    assert explanations["global_importances"]["x"] == pytest.approx(1.0, rel=1e-6)
    assert explanations["local_explanations"][0]["impact"] == pytest.approx(1.0, rel=1e-6)
    assert len(calls) == 3


def test_predict_skips_explanations_for_non_numeric_payload() -> None:
    ns = _make_prediction_namespace()

    def runner(model_name: str, instance: Any, payload: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        return {"score": 0.5}

    model_name = _configure_model(ns, runner)
    predict = ns["predict"]

    result = predict(model_name, {"text": "hello"})

    assert result["status"] == "ok"
    assert "explanations" not in result


def test_call_python_model_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _make_registry_namespace()
    module_path = tmp_path / "demo_module.py"
    module_path.write_text(
        "def predict(value=None):\n"
        "    return {'echo': value}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    response = ns["call_python_model"]("demo_module", "predict", {"value": 7})

    assert response["status"] == "ok"
    assert response["result"]["echo"] == 7


def test_call_python_model_error_without_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _make_registry_namespace()
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    response = ns["call_python_model"]("missing.module", "predict", {})

    assert response["status"] == "error"
    assert "traceback" in response and response["traceback"]
    assert response["error"].startswith("ImportError") or "Module" in response["error"]


def test_call_python_model_stub_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _make_registry_namespace()
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "1")

    response = ns["call_python_model"]("missing.stub", "predict", {})

    assert response["status"] == "stub"
    assert response["result"] == "stub_prediction"
    assert "error" in response