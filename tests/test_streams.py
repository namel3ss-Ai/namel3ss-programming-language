from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

from namel3ss.codegen.backend.core import generate_backend
from namel3ss.parser import Parser


def _load_backend_modules(package_name: str, backend_dir: Path) -> tuple[object, object]:
    init_py = backend_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_py,
        submodule_search_locations=[str(backend_dir)],
    )
    assert spec and spec.loader, "failed to load backend package"
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    main_module = importlib.import_module(f"{package_name}.main")
    runtime_module = importlib.import_module(f"{package_name}.generated.runtime")
    return main_module, runtime_module


def _write_backend(tmp_path: Path) -> Path:
    source = (
        'app "Realtime".\n'
        '\n'
        'dataset "events" from table events:\n'
        '  cache:\n'
        '    strategy: memory\n'
        '\n'
        'page "Dashboard" at "/dashboard":\n'
        '  show text "Hello"\n'
    )
    app = Parser(source).parse_app()
    backend_dir = tmp_path / "backend_stream"
    generate_backend(app, backend_dir, enable_realtime=True)
    return backend_dir


def test_page_stream_replays_cached_event(tmp_path: Path) -> None:
    test_client_module = pytest.importorskip("fastapi.testclient")
    TestClient = test_client_module.TestClient
    backend_dir = _write_backend(tmp_path)
    package_name = "backend_stream_pkg"
    try:
        main_module, runtime_module = _load_backend_modules(package_name, backend_dir)
        pages_module = importlib.import_module(f"{package_name}.generated.routers.pages")
        pages_source = Path(pages_module.__file__).read_text(encoding="utf-8")
        assert "stream_page_events" in pages_source, pages_source
        paths = {route.path for route in main_module.app.routes}
        assert "/api/streams/pages/{slug}" in paths
        asyncio.run(runtime_module.publish_event("page::dashboard", {"message": "hello"}))
        with TestClient(main_module.app) as client:
            with client.stream("GET", "/api/streams/pages/dashboard") as response:
                line_iter = response.iter_lines()
                first_line = next(line_iter)
                while first_line and first_line.startswith(":"):
                    first_line = next(line_iter)
                assert first_line.startswith("data:"), first_line
                payload = json.loads(first_line.split("data:", 1)[1].strip())
                assert payload["message"] == "hello"
    finally:
        removal_prefixes = [package_name, f"{package_name}."]
        to_remove = [name for name in sys.modules if any(name == prefix or name.startswith(prefix) for prefix in removal_prefixes)]
        for name in to_remove:
            sys.modules.pop(name, None)
