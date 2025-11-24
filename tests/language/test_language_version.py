import re
from pathlib import Path

from namel3ss.lang import LANGUAGE_VERSION, SUPPORTED_LANGUAGE_VERSIONS


def test_language_version_semver_and_supported():
    assert re.fullmatch(r"\d+\.\d+\.\d+", LANGUAGE_VERSION)
    assert LANGUAGE_VERSION in SUPPORTED_LANGUAGE_VERSIONS


def test_language_spec_document_matches_version():
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "docs" / "spec" / f"namel3ss-language-spec-v{LANGUAGE_VERSION}.md"
    assert spec_path.exists(), f"Missing language specification document: {spec_path}"
    contents = spec_path.read_text(encoding="utf-8")
    assert LANGUAGE_VERSION in contents
