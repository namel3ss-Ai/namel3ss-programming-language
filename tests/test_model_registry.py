import json
from pathlib import Path

from namel3ss.ml.registry import (
    MODEL_REGISTRY_ENV,
    get_default_model_registry,
    load_model_registry,
)


def test_load_model_registry_merges_environment_file(tmp_path, monkeypatch):
    overrides = {
        "image_classifier": {
            "metadata": {"threshold": 0.75, "notes": "from_env"},
        },
        "new_model": {
            "type": "xgboost",
            "framework": "xgboost",
            "version": "v2",
            "metrics": {"accuracy": 0.88},
            "metadata": {"loader": "pkg.module:load_model"},
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(overrides), encoding="utf-8")

    monkeypatch.setenv(MODEL_REGISTRY_ENV, str(registry_path))

    merged = load_model_registry()

    assert merged["image_classifier"]["metadata"]["threshold"] == 0.75
    assert merged["image_classifier"]["metadata"]["notes"] == "from_env"
    assert merged["new_model"]["framework"] == "xgboost"
    assert merged["churn_classifier"]["metadata"]["loader"].startswith("namel3ss.ml.hooks")

    default_registry = get_default_model_registry()
    assert default_registry["image_classifier"]["metadata"].get("threshold") == 0.5


def test_load_model_registry_accepts_inline_json(monkeypatch):
    override_json = json.dumps(
        {
            "image_classifier": {
                "metadata": {"threshold": 0.9},
            }
        }
    )
    monkeypatch.setenv(MODEL_REGISTRY_ENV, override_json)

    merged = load_model_registry()

    assert merged["image_classifier"]["metadata"]["threshold"] == 0.9
    assert merged["churn_classifier"]["metadata"]["intercept"] == -0.2

    monkeypatch.delenv(MODEL_REGISTRY_ENV, raising=False)

    restored = load_model_registry()
    assert restored["image_classifier"]["metadata"]["threshold"] == 0.5
