import pytest

from namel3ss.ast.program import Program
from namel3ss.parser import Parser
from namel3ss.resolver import ModuleResolutionError, resolve_program


def _parse_module(source: str, path: str):
    module = Parser(source, path=path).parse()
    if not module.name:
        module.name = path
    module.path = path
    return module


def test_mixed_language_versions_error(monkeypatch):
    monkeypatch.setattr("namel3ss.resolver.SUPPORTED_LANGUAGE_VERSIONS", ["0.1.0", "0.2.0"])
    mod_a = _parse_module('language_version "0.1.0"\nmodule a\napp "A".', "a.n3")
    mod_b = _parse_module('language_version "0.2.0"\nmodule b\napp "B".', "b.n3")
    program = Program(modules=[mod_a, mod_b])
    with pytest.raises(ModuleResolutionError):
        resolve_program(program)


def test_unsupported_language_version_error():
    module = _parse_module('language_version "9.9.9"\nmodule test\napp "A".', "test.n3")
    program = Program(modules=[module])
    with pytest.raises(ModuleResolutionError):
        resolve_program(program)


def test_unspecified_language_defaults_to_current():
    module = _parse_module('module test\napp "A".', "test.n3")
    program = Program(modules=[module])
    resolved = resolve_program(program)
    assert resolved.language_version is not None
