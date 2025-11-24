from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from namel3ss.codegen.backend.core import generate_backend
from namel3ss.parser import Parser


def _build_frames_app_source() -> str:
    return (
        'app "FramesApp".\n'
        '\n'
        'frame "UsersFrame" from table users:\n'
        '  column id number required\n'
        '  column name string\n'
        '  example:\n'
        '    id: 1\n'
        '    name: "Alice"\n'
        '  example:\n'
        '    id: 2\n'
        '    name: "Bob"\n'
        '\n'
        'dataset "users" from table users:\n'
        '  add column id = 1\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Hello"\n'
    )


def _generate_frames_backend(tmp_path: Path) -> Path:
    app = Parser(_build_frames_app_source()).parse_app()
    backend_dir = tmp_path / "frames_backend"
    generate_backend(app, backend_dir)
    return backend_dir


def _load_backend_modules(package_name: str, backend_dir: Path):
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
    return main_module


def _cleanup_modules(package_name: str) -> None:
    removal_prefixes = [package_name, f"{package_name}."]
    to_remove = [name for name in sys.modules if any(name == prefix or name.startswith(prefix) for prefix in removal_prefixes)]
    for name in to_remove:
        sys.modules.pop(name, None)


def test_frames_router_endpoints(tmp_path: Path) -> None:
    test_client_module = pytest.importorskip("fastapi.testclient")
    TestClient = test_client_module.TestClient
    backend_dir = _generate_frames_backend(tmp_path)
    package_name = "frames_backend_pkg"
    try:
        main_module = _load_backend_modules(package_name, backend_dir)
        with TestClient(main_module.app) as client:
            list_response = client.get("/api/frames")
            assert list_response.status_code == 200
            frames = list_response.json()
            assert "UsersFrame" in frames

            detail_response = client.get(
                "/api/frames/UsersFrame",
                params={"limit": 1, "offset": 0, "order_by": "name:asc"},
            )
            assert detail_response.status_code == 200
            detail_payload = detail_response.json()
            assert detail_payload["name"] == "UsersFrame"
            assert detail_payload["schema"]["columns"]
            assert detail_payload["rows"]
            assert detail_payload["limit"] == 1
            assert detail_payload["total"] >= 2

            schema_response = client.get("/api/frames/UsersFrame/schema")
            assert schema_response.status_code == 200
            schema_payload = schema_response.json()
            assert schema_payload["columns"]
            assert schema_payload["columns"][0]["name"] == "id"

            csv_response = client.get("/api/frames/UsersFrame.csv")
            assert csv_response.status_code == 200
            assert csv_response.headers.get("content-type", "").startswith("text/csv")
            assert csv_response.text.strip()

            parquet_response = client.get("/api/frames/UsersFrame.parquet")
            assert parquet_response.status_code == 200
            assert parquet_response.content
    finally:
        _cleanup_modules(package_name)
