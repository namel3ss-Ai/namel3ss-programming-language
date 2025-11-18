from pathlib import Path

import pytest

from namel3ss.ast import App, Import, ImportedName
from namel3ss.loader import extract_single_app, load_program
from namel3ss.resolver import ModuleResolutionError, resolve_program


def _write_app(path: Path, name: str) -> None:
    path.write_text(
        f'app "{name}".\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n',
        encoding="utf-8",
    )


def test_load_program_from_file(tmp_path: Path) -> None:
    source = tmp_path / "test_app.n3"
    _write_app(source, "Test App")

    program = load_program(source)

    assert len(program.modules) == 1
    module = program.modules[0]
    assert module.path == str(source.resolve())
    assert module.body
    app = module.body[0]
    assert isinstance(app, App)
    assert app.name == "Test App"
    loaded_app = extract_single_app(program)
    assert loaded_app.name == "Test App"


def test_load_program_from_directory(tmp_path: Path) -> None:
    first = tmp_path / "first.n3"
    second = tmp_path / "nested" / "second.n3"
    second.parent.mkdir(parents=True, exist_ok=True)
    _write_app(first, "First")
    _write_app(second, "Second")

    program = load_program(tmp_path)

    assert len(program.modules) == 2
    module_paths = [module.path for module in program.modules]
    expected_paths = [str(first.resolve()), str(second.resolve())]
    assert module_paths == sorted(expected_paths)
    module_names = sorted(module.name for module in program.modules)
    assert module_names == ["first", "nested.second"]


def test_extract_single_app_errors_for_multiple_modules(tmp_path: Path) -> None:
    first = tmp_path / "one.n3"
    second = tmp_path / "two.n3"
    second.parent.mkdir(parents=True, exist_ok=True)
    _write_app(first, "One")
    _write_app(second, "Two")
    program = load_program(tmp_path)

    with pytest.raises(ModuleResolutionError):
        extract_single_app(program)


def test_load_program_preserves_module_metadata(tmp_path: Path) -> None:
    source = tmp_path / "module_app.n3"
    source.write_text(
        'module my_app.billing\n'
        'import lib.tools as tools\n'
        'import lib.templates: Layout, Invoice as InvoiceTemplate\n'
        'app "Billing".\n'
        'page "Home" at "/":\n'
        '  show text "hello"\n',
        encoding="utf-8",
    )

    program = load_program(source)

    assert len(program.modules) == 1
    module = program.modules[0]
    assert module.name == "my_app.billing"
    assert module.imports == [
        Import(module="lib.tools", names=None, alias="tools"),
        Import(
            module="lib.templates",
            names=[
                ImportedName(name="Layout", alias=None),
                ImportedName(name="Invoice", alias="InvoiceTemplate"),
            ],
            alias=None,
        ),
    ]


def test_resolve_program_derives_app_from_single_file(tmp_path: Path) -> None:
    source = tmp_path / "app.n3"
    _write_app(source, "Root App")

    program = load_program(source)
    resolved = resolve_program(program, entry_path=source)
    assert resolved.app.name == "Root App"
