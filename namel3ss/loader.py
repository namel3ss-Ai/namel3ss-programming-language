"""Utilities for loading .n3 source trees into Program ASTs."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, List

from namel3ss.ast import App, Module, Program
from namel3ss.parser import Parser
from namel3ss.resolver import resolve_program


def _derive_module_name(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        parts = [path.stem]
    else:
        parts = list(relative.parts)
        if parts:
            parts[-1] = Path(parts[-1]).stem
    name = ".".join(part for part in parts if part)
    return name or path.stem


def _discover_source_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".n3" else []
    return sorted(path for path in root.rglob("*.n3") if path.is_file())


def _iter_source_files(root: Path) -> Iterator[Path]:
    for path in _discover_source_files(root):
        yield path


def _parse_module(source_path: Path) -> Module:
    text = source_path.read_text(encoding="utf-8")
    parser = Parser(text, path=str(source_path))
    module = parser.parse()
    module.path = str(source_path)
    return module


def load_program(root_path: str | PathLike[str]) -> Program:
    root = Path(root_path).resolve()
    project_root = root if root.is_dir() else root.parent
    module_paths: List[Path] = list(_iter_source_files(root))
    if not module_paths:
        raise FileNotFoundError(f"No .n3 files found at {root}")
    modules: List[Module] = []
    for path in module_paths:
        module = _parse_module(path)
        if not module.name:
            module.name = _derive_module_name(path, project_root)
        modules.append(module)
    return Program(modules=modules)


def extract_single_app(program: Program) -> App:
    resolved = resolve_program(program)
    return resolved.app


__all__ = ["load_program", "extract_single_app"]
