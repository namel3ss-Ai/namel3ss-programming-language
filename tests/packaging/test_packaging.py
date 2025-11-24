"""Packaging metadata sanity checks."""

from __future__ import annotations

from pathlib import Path

import namel3ss

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore[assignment]


_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_pyproject_version_matches_package() -> None:
    metadata = _load_pyproject()["project"]
    assert metadata["version"] == namel3ss.__version__


def test_expected_extras_are_declared() -> None:
    metadata = _load_pyproject()["project"]
    extras = metadata.get("optional-dependencies", {})
    expected = {
        "cli",
        "sql",
        "redis",
        "mongo",
        "realtime",
        "ai-connectors",
        "observability",
        "dev",
        "all",
    }
    assert expected.issubset(extras.keys())
    dev_reqs = extras["dev"]
    assert any(req.startswith("pytest") for req in dev_reqs)
    observability_reqs = extras["observability"]
    assert any(req.startswith("opentelemetry") for req in observability_reqs)
