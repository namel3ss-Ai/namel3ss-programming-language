"""Runtime tests for experiment and frame integration."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser

from tests.backend_test_utils import load_backend_module


EXPERIMENT_SOURCE = (
    'app "ML".\n'
    '\n'
    'frame "ChurnFrame":\n'
    '  column user_id string:\n'
    '    role: id\n'
    '  column tenure number:\n'
    '    role: feature\n'
    '  column spend number:\n'
    '    role: feature\n'
    '  column churned bool:\n'
    '    role: target\n'
    '  column observed_time datetime:\n'
    '    role: time\n'
    '  splits:\n'
    '    train: 0.5\n'
    '    test: 0.5\n'
    '  sample:\n'
    '    user_id: "u1"\n'
    '    tenure: 5\n'
    '    spend: 120\n'
    '    churned: false\n'
    '    observed_time: "2024-01-01"\n'
    '  sample:\n'
    '    user_id: "u2"\n'
    '    tenure: 15\n'
    '    spend: 140\n'
    '    churned: true\n'
    '    observed_time: "2024-01-02"\n'
    '  sample:\n'
    '    user_id: "u3"\n'
    '    tenure: 20\n'
    '    spend: 60\n'
    '    churned: true\n'
    '    observed_time: "2024-01-03"\n'
    '  sample:\n'
    '    user_id: "u4"\n'
    '    tenure: 8\n'
    '    spend: 90\n'
    '    churned: false\n'
    '    observed_time: "2024-01-04"\n'
    '\n'
    'experiment "churn_eval":\n'
    '  metadata:\n'
    '    data:\n'
    '      frame: ChurnFrame\n'
    '      target: churned\n'
    '      features: tenure, spend\n'
    '  variants:\n'
    '    dataset_variant uses model baseline\n'
    '  metrics:\n'
    '    accuracy\n'
    '\n'
    'experiment "churn_invalid":\n'
    '  metadata:\n'
    '    data:\n'
    '      frame: ChurnFrame\n'
    '      target: missing_column\n'
    '  variants:\n'
    '    invalid_variant uses model baseline\n'
)


def _build_backend(tmp_path):
    app = Parser(EXPERIMENT_SOURCE).parse_app()
    for experiment in app.experiments:
        for variant in experiment.variants:
            variant.target_type = "python"
            variant.target_name = "tests.runtime_plugins"
            variant.config["method"] = "dataset_model"
    backend_dir = tmp_path / "generated"
    generate_backend(app, backend_dir)
    return backend_dir


def test_experiment_evaluates_with_frame_dataset(tmp_path, monkeypatch):
    backend_dir = _build_backend(tmp_path)
    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime = module.runtime
        result = runtime.evaluate_experiment("churn_eval", {})
        assert result["status"] == "ok"
        dataset = result["inputs"]["dataset"]
        assert dataset["frame"] == "ChurnFrame"
        assert dataset["features"][:2] == ["tenure", "spend"]
        assert dataset["target"] == "churned"
        assert dataset["splits"]["train"]["size"] == 2
        assert dataset["splits"]["test"]["size"] == 2
        variant_result = result["variants"][0]
        assert variant_result["metrics"]["accuracy"] == 1.0


def test_invalid_data_config_surfaces_error(tmp_path, monkeypatch):
    backend_dir = _build_backend(tmp_path)
    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime = module.runtime
        result = runtime.evaluate_experiment("churn_invalid", {})
        assert result["status"] == "error"
        assert result["error"] == "experiment_data_error"
        assert "missing_column" in str(result.get("detail"))
