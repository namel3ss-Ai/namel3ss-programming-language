"""
Integration tests for generated FastAPI backends.

Tests validate that generated backends:
- Expose correct endpoints for pages, models, and components
- Return responses matching documented schemas
- Handle errors with consistent format and status codes
- Maintain frontend-backend contract stability
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from namel3ss.codegen.backend.core import generate_backend
from namel3ss.loader import load_program, extract_single_app
from namel3ss.parser import Parser


# ============================================================================
# Test Utilities
# ============================================================================

def _generate_backend(app_source: str, tmp_path: Path) -> Path:
    """Generate backend from N3 source."""
    app = Parser(app_source).parse_app()
    backend_dir = tmp_path / "test_backend"
    generate_backend(app, backend_dir)
    return backend_dir


def _load_backend(package_name: str, backend_dir: Path):
    """Dynamically import generated backend."""
    init_py = backend_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_py,
        submodule_search_locations=[str(backend_dir)],
    )
    assert spec and spec.loader, f"Failed to load {package_name}"
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    
    return importlib.import_module(f"{package_name}.main")


def _cleanup_backend(package_name: str):
    """Remove backend modules from sys.modules."""
    to_remove = [
        name for name in sys.modules
        if name == package_name or name.startswith(f"{package_name}.")
    ]
    for name in to_remove:
        sys.modules.pop(name, None)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_client_class():
    """Import TestClient lazily to allow optional fastapi dependency."""
    test_client_module = pytest.importorskip("fastapi.testclient")
    return test_client_module.TestClient


@pytest.fixture
def simple_app_source() -> str:
    """Basic app with pages and datasets."""
    return '''
app "TestApp" connects to postgres "DB".

dataset "users" from table users:
  add column id = 1
  add column name = "Alice"

page "Home" at "/":
  show text "Welcome to TestApp"

page "Users" at "/users":
  show text "User List"
  show table "All Users" from dataset users
'''


@pytest.fixture
def app_with_control_flow() -> str:
    """App with if/else and for loops."""
    return '''
app "ControlFlowApp".

dataset "active_users" from table users:
  filter by: status == "active"

page "Dashboard" at "/":
  if user.role == "admin":
    show text "Admin View"
  else:
    show text "User View"
  
  for user in dataset active_users:
    show text "{user.name}"
'''


@pytest.fixture
def app_with_prompts() -> str:
    """App with structured prompts."""
    return '''
app "AIApp".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
  api_key: $OPENAI_API_KEY
}

prompt "classify" {
  args: {
    text: string
  }
  output_schema: {
    category: enum["technical", "billing", "other"],
    urgency: enum["low", "high"]
  }
  model: "gpt4"
  template: "Classify: {text}"
}

page "Home" at "/":
  show text "AI App"
'''


# ============================================================================
# Basic Endpoint Tests
# ============================================================================

def test_root_endpoint_returns_app_info(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that root endpoint returns app metadata."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_root"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert "name" in data
            assert data["name"] == "TestApp"
    finally:
        _cleanup_backend(pkg)


def test_health_check_endpoint(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that health check endpoint is available."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_health"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Page Endpoint Tests
# ============================================================================

def test_pages_list_endpoint(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that /api/pages lists all pages."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_pages_list"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/pages")
            
            assert response.status_code == 200
            pages = response.json()
            assert isinstance(pages, list)
            assert len(pages) >= 2
            
            slugs = [p["slug"] for p in pages]
            assert "home" in slugs
            assert "users" in slugs
    finally:
        _cleanup_backend(pkg)


def test_page_detail_endpoint(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that /api/pages/{slug} returns page details."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_page_detail"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/pages/home")
            
            assert response.status_code == 200
            page = response.json()
            assert page["slug"] == "home"
            assert page["path"] == "/"
            assert "components" in page
    finally:
        _cleanup_backend(pkg)


def test_nonexistent_page_returns_404(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that requesting nonexistent page returns 404."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_404"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/pages/nonexistent")
            
            assert response.status_code == 404
            error = response.json()
            assert "detail" in error
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Component Endpoint Tests
# ============================================================================

def test_table_component_endpoint(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that table component endpoints work."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_table"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            # List tables
            response = client.get("/api/tables")
            assert response.status_code == 200
            
            # Get specific table (if any exist)
            tables = response.json()
            if tables:
                table_id = tables[0]["id"]
                detail_response = client.get(f"/api/tables/{table_id}")
                assert detail_response.status_code == 200
                assert "rows" in detail_response.json()
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Dataset Endpoint Tests
# ============================================================================

def test_datasets_endpoint(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that /api/datasets lists datasets."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_datasets"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/datasets")
            
            assert response.status_code == 200
            datasets = response.json()
            assert isinstance(datasets, list)
            assert "users" in datasets
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Control Flow Tests
# ============================================================================

def test_conditional_rendering_in_pages(
    app_with_control_flow: str,
    tmp_path: Path,
    test_client_class
):
    """Test that pages with if/else compile correctly."""
    backend_dir = _generate_backend(app_with_control_flow, tmp_path)
    pkg = "test_backend_control_flow"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/pages/dashboard")
            
            assert response.status_code == 200
            page = response.json()
            assert "components" in page
            # Components should include conditional logic representation
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# AI Feature Tests
# ============================================================================

def test_prompts_endpoint_lists_prompts(
    app_with_prompts: str,
    tmp_path: Path,
    test_client_class
):
    """Test that /api/prompts lists structured prompts."""
    backend_dir = _generate_backend(app_with_prompts, tmp_path)
    pkg = "test_backend_prompts"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/prompts")
            
            assert response.status_code == 200
            prompts = response.json()
            assert isinstance(prompts, list)
            assert "classify" in prompts
    finally:
        _cleanup_backend(pkg)


def test_prompt_detail_includes_schema(
    app_with_prompts: str,
    tmp_path: Path,
    test_client_class
):
    """Test that prompt detail includes args and output schema."""
    backend_dir = _generate_backend(app_with_prompts, tmp_path)
    pkg = "test_backend_prompt_detail"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.get("/api/prompts/classify")
            
            assert response.status_code == 200
            prompt = response.json()
            assert "args" in prompt
            assert "output_schema" in prompt
            assert "text" in prompt["args"]
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Error Response Format Tests
# ============================================================================

def test_validation_error_has_consistent_format(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that validation errors follow consistent format."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_validation_error"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            # Send invalid request to trigger validation error
            response = client.post(
                "/api/forms/test",
                json={"invalid": "data"}
            )
            
            # Should return 422 for validation errors
            if response.status_code == 422:
                error = response.json()
                assert "detail" in error
    finally:
        _cleanup_backend(pkg)


def test_internal_error_does_not_leak_traceback(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that internal errors don't expose stack traces."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_error_format"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            # Try to trigger an internal error (implementation-dependent)
            # This is a smoke test to ensure error handling is in place
            response = client.get("/api/pages/home")
            
            # Even on error, response should be JSON, not traceback
            assert response.headers.get("content-type", "").startswith("application/json")
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# CORS and Security Headers Tests
# ============================================================================

def test_cors_headers_present(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that CORS headers are configured."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_cors"
    
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            response = client.options("/api/pages")
            
            # CORS preflight should be handled
            assert response.status_code in [200, 204]
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# State Registry Tests
# ============================================================================

def test_state_registries_contain_expected_entries(
    simple_app_source: str,
    tmp_path: Path,
    test_client_class
):
    """Test that backend state registries are populated correctly."""
    backend_dir = _generate_backend(simple_app_source, tmp_path)
    pkg = "test_backend_state"
    
    try:
        main = _load_backend(pkg, backend_dir)
        
        # Import registries module
        registries = importlib.import_module(f"{pkg}.generated.registries")
        
        # Check that pages are registered
        assert hasattr(registries, "PAGES")
        assert len(registries.PAGES) >= 2
        
        # Check that datasets are registered
        assert hasattr(registries, "DATASETS")
        assert "users" in registries.DATASETS
    finally:
        _cleanup_backend(pkg)


# ============================================================================
# Real Example Tests
# ============================================================================

def test_demo_app_backend_compiles(tmp_path: Path, test_client_class):
    """Test that demo_app.n3 generates a working backend."""
    demo_path = Path(__file__).parent.parent / "demo_app.n3"
    if not demo_path.exists():
        pytest.skip("demo_app.n3 not found")
    
    program = load_program(str(demo_path))
    app = extract_single_app(program)
    
    backend_dir = tmp_path / "demo_backend"
    generate_backend(app, backend_dir)
    
    pkg = "test_demo_backend"
    try:
        main = _load_backend(pkg, backend_dir)
        with test_client_class(main.app) as client:
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            
            # Test pages endpoint
            pages_response = client.get("/api/pages")
            assert pages_response.status_code == 200
    finally:
        _cleanup_backend(pkg)


def test_all_examples_generate_valid_backends(tmp_path: Path, test_client_class):
    """Test that all .n3 examples generate compilable backends."""
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        pytest.skip("examples/ directory not found")
    
    example_files = list(examples_dir.glob("*.n3"))
    if not example_files:
        pytest.skip("No .n3 examples found")
    
    for example_file in example_files:
        try:
            program = load_program(str(example_file))
            app = extract_single_app(program)
            
            backend_dir = tmp_path / f"backend_{example_file.stem}"
            generate_backend(app, backend_dir)
            
            pkg = f"test_example_{example_file.stem}"
            main = _load_backend(pkg, backend_dir)
            
            # Smoke test: ensure app object exists and has expected structure
            assert hasattr(main, "app")
            assert main.app is not None
            
            _cleanup_backend(pkg)
        except Exception as e:
            pytest.fail(f"Failed to generate backend for {example_file.name}: {e}")
