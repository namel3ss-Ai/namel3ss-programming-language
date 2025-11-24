"""Unit tests covering the deterministic ML hooks."""

from __future__ import annotations

import importlib
from typing import Any, Dict

import pytest


def _reload_hooks(monkeypatch: pytest.MonkeyPatch, allow_stubs: str | None) -> Any:
    if allow_stubs is None:
        monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    else:
        monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", allow_stubs)
    import namel3ss.ml.hooks as hooks

    return importlib.reload(hooks)


def test_image_hooks_require_driver_without_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, None)

    result = hooks.load_image_classifier({})

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["error"] == "image_model_not_configured"
    assert result["detail"] == "No image model driver is configured."


def test_image_hooks_require_driver_even_with_stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, "1")

    result = hooks.load_image_classifier({})

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["error"] == "image_model_not_configured"


def test_image_hooks_real_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, None)

    class FakeImageModel:
        def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {"score": 0.83, "label": "cat", "scores": {"cat": 0.83, "dog": 0.17}}

    model = FakeImageModel()
    payload = {"foo": 1.0}
    prediction = hooks.run_image_classifier(model, payload, demo=False)
    assert prediction["status"] == "ok"
    output = prediction["output"]
    assert output["score"] == pytest.approx(0.83)
    assert output["label"] == "cat"
    assert output["scores"]["cat"] == pytest.approx(0.83)

    explanation = hooks.explain_image_classifier(model, payload, prediction, demo=False)
    assert explanation is not None
    assert explanation["status"] == "ok"
    assert explanation["baseline_score"] == pytest.approx(0.83)
    locals_list = explanation["explanations"]["local_explanations"]
    assert locals_list[0]["feature"] == "foo"
    assert locals_list[0]["impact"] == pytest.approx(0.0, abs=1e-6)


def test_churn_hooks_require_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, None)

    result = hooks.load_churn_classifier({})

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["error"] == "churn_model_not_configured"
    assert result["detail"] == "No churn model driver is configured."


def test_churn_hooks_real_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, None)

    class FakeChurnModel:
        def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {"score": 0.42, "label": "retained"}

    model = FakeChurnModel()
    payload = {"tenure": 5.0}
    prediction = hooks.run_churn_classifier(model, payload, demo=False)
    assert prediction["status"] == "ok"
    assert prediction["output"]["score"] == pytest.approx(0.42)
    assert prediction["output"]["label"] == "retained"

    explanation = hooks.explain_churn_classifier(model, payload, prediction, demo=False)
    assert explanation is not None
    assert explanation["status"] == "ok"
    assert explanation["baseline_score"] == pytest.approx(0.42)


def test_numeric_perturbation_explanations(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = _reload_hooks(monkeypatch, None)

    class LinearModel:
        def __init__(self) -> None:
            self.coeffs = {"f1": 0.5, "f2": 2.0}

        def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            score = sum(self.coeffs[name] * inputs.get(name, 0.0) for name in self.coeffs)
            return {"score": score}

    model = LinearModel()
    payload = {"f1": 1.0, "f2": 0.25}

    prediction = hooks.run_image_classifier(model, payload)
    assert prediction["status"] == "ok"

    explanations = hooks.explain_image_classifier(model, payload, prediction)
    assert explanations is not None
    assert explanations["status"] == "ok"
    local = {item["feature"]: item["impact"] for item in explanations["explanations"]["local_explanations"]}
    assert local["f1"] == pytest.approx(0.5, abs=1e-6)
    assert local["f2"] == pytest.approx(2.0, abs=1e-6)
    global_imp = explanations["explanations"]["global_importances"]
    assert global_imp["f1"] == pytest.approx(0.2, abs=1e-6)
    assert global_imp["f2"] == pytest.approx(0.8, abs=1e-6)