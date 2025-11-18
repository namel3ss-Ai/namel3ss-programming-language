import textwrap

import pytest

from namel3ss.ast import Import, ImportedName
from namel3ss.parser import N3SyntaxError, Parser


def _parse_module(source: str):
    return Parser(textwrap.dedent(source)).parse()


def _app_body() -> str:
    return textwrap.dedent(
        """
        app "Example".

        page "Home" at "/":
          show text "hi"
        """
    )


def test_module_declaration_sets_module_name():
    module = _parse_module("module my_app.billing\n" + _app_body())
    assert module.name == "my_app.billing"
    assert module.imports == []


def test_module_declaration_optional():
    module = _parse_module(_app_body())
    assert module.name is None


def test_multiple_module_declarations_raise():
    source = "module first\nmodule second\n" + _app_body()
    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_module_declaration_must_precede_imports():
    source = "import shared.utils\nmodule example.core\n" + _app_body()
    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_import_statements_parse_all_forms():
    source = textwrap.dedent(
        """
        module my_app.billing
        import my_app.shared.templates
        import my_app.models as models
        import my_app.shared.templates: OrderSummary, InvoiceEmail as InvoiceSummary
        """
    ) + _app_body()

    module = _parse_module(source)

    assert module.imports == [
        Import(module="my_app.shared.templates", names=None, alias=None),
        Import(module="my_app.models", names=None, alias="models"),
        Import(
            module="my_app.shared.templates",
            names=[
                ImportedName(name="OrderSummary", alias=None),
                ImportedName(name="InvoiceEmail", alias="InvoiceSummary"),
            ],
            alias=None,
        ),
    ]


def test_import_alias_not_allowed_with_selective_names():
    source = textwrap.dedent(
        """
        import lib.templates as tmpl: Card
        """
    ) + _app_body()

    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_imports_must_precede_declarations():
    source = _app_body() + "\nimport my_app.extra"
    with pytest.raises(N3SyntaxError):
        _parse_module(source)


def test_module_without_app_allows_definitions():
    source = textwrap.dedent(
        """
        module support.pages
        page "About" at "/about":
          show text "about"
        """
    )
    module = _parse_module(source)
    assert module.body
    assert not module.has_explicit_app
