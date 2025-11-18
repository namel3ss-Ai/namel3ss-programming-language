from namel3ss.ast import Module, App
from namel3ss.parser import Parser


def test_parser_returns_module_with_app_body() -> None:
    source = (
        'app "Sample".\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hello"\n'
    )

    parser = Parser(source, path="sample.n3")
    module = parser.parse()

    assert isinstance(module, Module)
    assert module.path == "sample.n3"
    assert len(module.body) == 1
    assert isinstance(module.body[0], App)
