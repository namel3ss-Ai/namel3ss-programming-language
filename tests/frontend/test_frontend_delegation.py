"""Tests ensuring the static frontend generator reuses shared assets."""

from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace


FRONTEND_MODULE_PATH = Path(__file__).resolve().parents[1] / "namel3ss" / "codegen" / "frontend.py"

spec = importlib_util.spec_from_file_location("namel3ss.codegen._static_frontend", FRONTEND_MODULE_PATH)
assert spec and spec.loader, "Failed to locate static frontend generator module"
frontend = importlib_util.module_from_spec(spec)
spec.loader.exec_module(frontend)


def test_generate_styles_delegates(monkeypatch):
    sentinel_app = SimpleNamespace(theme=SimpleNamespace(values={"primary": "#123456"}))
    calls = {}

    def fake_styles(app):
        calls["app"] = app
        return "/*css*/"

    monkeypatch.setattr(frontend, "_assets_generate_styles", fake_styles)

    result = frontend._generate_styles(sentinel_app)

    assert result == "/*css*/"
    assert calls["app"] is sentinel_app


def test_generate_widget_library_delegates(monkeypatch):
    calls = {}

    def fake_widgets():
        calls["invoked"] = True
        return "//js"

    monkeypatch.setattr(frontend, "_assets_generate_widget_library", fake_widgets)

    result = frontend._generate_widget_library()

    assert result == "//js"
    assert calls["invoked"]
