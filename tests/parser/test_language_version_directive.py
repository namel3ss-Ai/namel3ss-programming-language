import pytest

from namel3ss.parser import Parser, N3SyntaxError


def test_language_version_directive_sets_module_field():
    source = 'language_version "0.1.0"\napp "Example".'
    module = Parser(source).parse()
    assert module.language_version == "0.1.0"


def test_multiple_language_version_directives_error():
    source = 'language_version "0.1.0"\nlanguage_version "0.1.1"\napp "Example".'
    with pytest.raises(N3SyntaxError):
        Parser(source).parse()


def test_language_version_must_precede_body():
    source = 'app "Example".\nlanguage_version "0.1.0"'
    with pytest.raises(N3SyntaxError):
        Parser(source).parse()
