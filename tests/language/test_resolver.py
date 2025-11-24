from pathlib import Path

import pytest

from namel3ss.loader import load_program
from namel3ss.resolver import ModuleResolutionError, resolve_program


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolver_merges_supporting_modules(tmp_path: Path) -> None:
    root = tmp_path / "app.n3"
    support = tmp_path / "shared" / "pages.n3"
    _write(
        root,
        'module multi.app\n'
        'import multi.shared.pages\n'
        'app "Root".\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "home"\n',
    )
    _write(
        support,
        'module multi.shared.pages\n'
        '\n'
        'page "Help" at "/help":\n'
        '  show text "help"\n',
    )

    program = load_program(tmp_path)
    resolved = resolve_program(program, entry_path=root)

    page_names = [page.name for page in resolved.app.pages]
    assert sorted(page_names) == ["Help", "Home"]


def test_resolver_requires_single_root_app(tmp_path: Path) -> None:
    module_path = tmp_path / "support.n3"
    _write(
        module_path,
        'module lone.support\n'
        '\n'
        'page "Solo" at "/solo":\n'
        '  show text "solo"\n',
    )
    program = load_program(tmp_path)
    with pytest.raises(ModuleResolutionError):
        resolve_program(program, entry_path=module_path)


def test_resolver_rejects_multiple_roots(tmp_path: Path) -> None:
    first = tmp_path / "one.n3"
    second = tmp_path / "two.n3"
    _write(
        first,
        'module multi.one\n'
        'app "One".\n'
        'page "First" at "/":\n'
        '  show text "one"\n',
    )
    _write(
        second,
        'module multi.two\n'
        'app "Two".\n'
        'page "Second" at "/two":\n'
        '  show text "two"\n',
    )
    program = load_program(tmp_path)
    with pytest.raises(ModuleResolutionError):
        resolve_program(program, entry_path=first)


def test_resolver_validates_imported_symbols(tmp_path: Path) -> None:
    root = tmp_path / "app.n3"
    shared = tmp_path / "templates.n3"
    _write(
        root,
        'module import.root\n'
        'import import.shared: MissingSymbol\n'
        'app "ImportRoot".\n'
        'page "Home" at "/":\n'
        '  show text "home"\n',
    )
    _write(
        shared,
        'module import.shared\n'
        '\n'
        'define template "Invoice":\n'
        '  prompt = "hi"\n',
    )
    program = load_program(tmp_path)
    with pytest.raises(ModuleResolutionError):
        resolve_program(program, entry_path=root)


def test_resolver_rejects_unknown_training_job_reference(tmp_path: Path) -> None:
    root = tmp_path / "app.n3"
    _write(
        root,
        'module training.app\n'
        '\n'
        'app "Trainer".\n'
        '\n'
        'dataset "training_data" from table training_table.\n'
        '\n'
        'model "image_classifier" using openai:\n'
        '  name: "gpt-4o-mini"\n'
        '\n'
        'training "baseline":\n'
        '  model: "image_classifier"\n'
        '  dataset: "training_data"\n'
        '  objective: "maximize_accuracy"\n'
        '\n'
        'tuning "grid":\n'
        '  training_job: "missing"\n'
        '  search_space:\n'
        '    learning_rate:\n'
        '      type: float\n'
        '      min: 0.001\n'
        '      max: 0.01\n'
    )

    program = load_program(tmp_path)
    with pytest.raises(ModuleResolutionError) as excinfo:
        resolve_program(program, entry_path=root)

    assert "unknown training job" in str(excinfo.value)
