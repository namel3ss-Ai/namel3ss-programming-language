"""Tests for the Namel3ss CLI."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from namel3ss.cli import (
    _apply_env_overrides,
    _format_error_detail,
    _traceback_excerpt,
    cmd_build,
    cmd_deploy,
    cmd_doctor,
    cmd_run,
    cmd_train,
    main,
    prepare_backend,
    run_dev_server,
    check_uvicorn_available,
)
from namel3ss.parser import N3SyntaxError


# Sample N3 code for testing
SAMPLE_N3 = '''app "Test App" connects to postgres "TEST_DB".

page "Home" at "/":
  show text "Hello World"
'''


@pytest.fixture
def temp_n3_file(tmp_path):
    """Create a temporary .n3 file for testing."""
    n3_file = tmp_path / "test_app.n3"
    n3_file.write_text(SAMPLE_N3)
    return n3_file


@pytest.fixture
def invalid_n3_file(tmp_path):
    """Create an invalid .n3 file for testing."""
    n3_file = tmp_path / "invalid.n3"
    n3_file.write_text('app "Test".\npage "Home" at "/":\n  invalid syntax here')
    return n3_file


def test_format_error_detail_truncates_long_messages():
    class CustomError(RuntimeError):
        pass

    message = "x" * 500
    detail = _format_error_detail(CustomError(message))

    assert len(detail) <= 280
    assert detail.endswith("...")


def test_traceback_excerpt_limits_trace(monkeypatch):
    try:
        raise ValueError("boom")
    except ValueError:
        excerpt = _traceback_excerpt()

    assert "ValueError" in excerpt
    assert len(excerpt) <= 4000


def test_prepare_backend_success(temp_n3_file, tmp_path):
    """Test that prepare_backend generates backend scaffold."""
    backend_dir = tmp_path / "backend"
    
    app = prepare_backend(temp_n3_file, str(backend_dir))
    
    assert app.name == "Test App"
    assert backend_dir.exists()
    # Check that key backend files were created
    assert (backend_dir / "main.py").exists()
    assert (backend_dir / "database.py").exists()


def test_prepare_backend_file_not_found():
    """Test that prepare_backend raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        prepare_backend(Path("nonexistent.n3"), "backend")


def test_prepare_backend_syntax_error(invalid_n3_file, tmp_path):
    """Test that prepare_backend raises N3SyntaxError for invalid syntax."""
    backend_dir = tmp_path / "backend"
    
    with pytest.raises(N3SyntaxError):
        prepare_backend(invalid_n3_file, str(backend_dir))


def test_check_uvicorn_available():
    """Test uvicorn availability check."""
    # This will return True or False depending on whether uvicorn is installed
    # We just check it doesn't crash
    result = check_uvicorn_available()
    assert isinstance(result, bool)


def test_cmd_build_static_only(temp_n3_file, tmp_path, capsys):
    """Test build command generates static site."""
    output_dir = tmp_path / "build"
    
    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.out = str(output_dir)
    args.print_ast = False
    args.backend_only = False
    args.build_backend = False
    args.backend_out = str(tmp_path / "backend")
    args.embed_insights = False
    
    cmd_build(args)
    
    # Check static files were created
    assert output_dir.exists()
    assert (output_dir / "styles.css").exists()
    
    # Check output message
    captured = capsys.readouterr()
    assert "Static site generated" in captured.out


def test_cmd_build_with_backend(temp_n3_file, tmp_path, capsys):
    """Test build command with --build-backend flag."""
    output_dir = tmp_path / "build"
    backend_dir = tmp_path / "backend"
    
    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.out = str(output_dir)
    args.print_ast = False
    args.backend_only = False
    args.build_backend = True
    args.backend_out = str(backend_dir)
    args.embed_insights = False
    
    cmd_build(args)
    
    # Check both static and backend were created
    assert output_dir.exists()
    assert backend_dir.exists()
    assert (backend_dir / "main.py").exists()
    
    captured = capsys.readouterr()
    assert "Static site generated" in captured.out
    assert "Backend scaffold generated" in captured.out
    assert "datasets available" in captured.out
    assert "connectors registered" in captured.out
    assert "insights routed" in captured.out


def test_cmd_build_backend_only(temp_n3_file, tmp_path, capsys):
    """Test build command with --backend-only flag."""
    backend_dir = tmp_path / "backend"
    
    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.out = str(tmp_path / "build")
    args.print_ast = False
    args.backend_only = True
    args.build_backend = False
    args.backend_out = str(backend_dir)
    args.embed_insights = False
    
    cmd_build(args)
    
    # Check only backend was created
    assert backend_dir.exists()
    assert (backend_dir / "main.py").exists()
    
    captured = capsys.readouterr()
    assert "Backend scaffold generated" in captured.out
    assert "Static site generated" not in captured.out
    assert "datasets available" in captured.out
    assert "connectors registered" in captured.out
    assert "insights routed" in captured.out


def test_cmd_build_applies_env_overrides(temp_n3_file, tmp_path):
    """Ensure env overrides passed via --env are applied."""
    backend_dir = tmp_path / "backend"

    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.out = str(tmp_path / "build")
    args.print_ast = False
    args.backend_only = True
    args.build_backend = False
    args.backend_out = str(backend_dir)
    args.embed_insights = False
    args.env = ["API_TOKEN=secret"]

    with mock.patch.dict(os.environ, {}, clear=True):
        cmd_build(args)
        assert os.environ.get("API_TOKEN") == "secret"

    # backend directory should still be produced
    assert backend_dir.exists()
    assert (backend_dir / "main.py").exists()


def test_cmd_build_print_ast(temp_n3_file, capsys):
    """Test build command with --print-ast flag."""
    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.print_ast = True
    args.backend_only = False
    args.build_backend = False
    args.embed_insights = False
    
    cmd_build(args)
    
    captured = capsys.readouterr()
    # Check AST was printed as JSON
    assert "Test App" in captured.out
    assert "name" in captured.out


def test_cmd_doctor_reports_optional_status(monkeypatch, capsys):
    """Doctor command should surface optional extras without failing."""

    from namel3ss.utils import dependencies as deps

    def fake_module_available(module: str) -> bool:
        available = {
            "fastapi": True,
            "httpx": True,
            "pydantic": True,
            "uvicorn": True,
            "sqlalchemy": False,
            "redis": False,
            "aioredis": False,
            "motor": False,
            "pymongo": False,
        }
        return available.get(module, False)

    monkeypatch.setattr(deps, "module_available", fake_module_available)

    args = argparse.Namespace()
    cmd_doctor(args)

    output = capsys.readouterr().out
    assert "✓ Core FastAPI runtime: available" in output
    assert "✗ Optional SQL connectors: missing" in output


def test_cmd_build_file_not_found(tmp_path, capsys):
    """Test build command with nonexistent file."""
    args = mock.Mock()
    args.env = []
    args.file = str(tmp_path / "nonexistent.n3")
    args.out = str(tmp_path / "build")
    args.print_ast = False
    args.backend_only = False
    args.build_backend = False
    args.embed_insights = False
    args.embed_insights = False
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_build(args)
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "file not found" in captured.err


def test_cmd_build_syntax_error(invalid_n3_file, tmp_path, capsys):
    """Test build command with syntax error in N3 file."""
    args = mock.Mock()
    args.env = []
    args.file = str(invalid_n3_file)
    args.out = str(tmp_path / "build")
    args.print_ast = False
    args.backend_only = False
    args.build_backend = False
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_build(args)
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Syntax error" in captured.err


@mock.patch('namel3ss.cli.check_uvicorn_available', return_value=True)
def test_run_dev_server_success(mock_check, temp_n3_file, tmp_path, capsys):
    """Test run_dev_server starts uvicorn correctly."""
    backend_dir = tmp_path / "backend"
    
    # Mock uvicorn module at import time
    mock_uvicorn = mock.Mock()
    mock_uvicorn.run = mock.Mock()
    
    # Patch sys.modules so the import inside run_dev_server gets our mock
    with mock.patch.dict('sys.modules', {'uvicorn': mock_uvicorn}):
        run_dev_server(
            temp_n3_file,
            backend_dir=str(backend_dir),
            host="127.0.0.1",
            port=8000,
            reload=True
        )
    
    # Check backend was generated
    assert backend_dir.exists()
    assert (backend_dir / "main.py").exists()
    
    # Check uvicorn.run was called with correct arguments
    mock_uvicorn.run.assert_called_once()
    call_args = mock_uvicorn.run.call_args
    assert call_args[1]['host'] == "127.0.0.1"
    assert call_args[1]['port'] == 8000
    assert call_args[1]['reload'] is True
    
    # Check output messages
    captured = capsys.readouterr()
    assert "Parsed: Test App" in captured.out
    assert "Backend generated" in captured.out
    assert "dev server running" in captured.out
    assert "datasets available" in captured.out
    assert "connectors registered" in captured.out
    assert "insights routed" in captured.out


@mock.patch('namel3ss.cli.check_uvicorn_available', return_value=False)
def test_run_dev_server_uvicorn_not_installed(mock_check, temp_n3_file, capsys):
    """Test run_dev_server fails gracefully when uvicorn is not installed."""
    with pytest.raises(SystemExit) as exc_info:
        run_dev_server(temp_n3_file)
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "uvicorn is not installed" in captured.err
    assert "pip install uvicorn" in captured.err


@mock.patch('namel3ss.cli.check_uvicorn_available', return_value=True)
def test_run_dev_server_syntax_error(mock_check, invalid_n3_file, capsys):
    """Test run_dev_server handles syntax errors."""
    # Mock uvicorn module
    mock_uvicorn = mock.Mock()
    
    with mock.patch.dict('sys.modules', {'uvicorn': mock_uvicorn}):
        with pytest.raises(SystemExit) as exc_info:
            run_dev_server(invalid_n3_file)
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Syntax error" in captured.err


@mock.patch('namel3ss.cli.check_uvicorn_available', return_value=True)
def test_cmd_run(mock_check, temp_n3_file, tmp_path):
    """Test cmd_run function."""
    mock_uvicorn = mock.Mock()
    mock_uvicorn.run = mock.Mock()
    
    args = mock.Mock()
    args.env = []
    args.file = str(temp_n3_file)
    args.backend_out = str(tmp_path / "backend")
    args.host = "0.0.0.0"
    args.port = 3000
    args.no_reload = True
    args.embed_insights = False
    
    with mock.patch.dict('sys.modules', {'uvicorn': mock_uvicorn}):
        cmd_run(args)
    
    # Check uvicorn was called with correct parameters
    mock_uvicorn.run.assert_called_once()
    call_args = mock_uvicorn.run.call_args
    assert call_args[1]['host'] == "0.0.0.0"
    assert call_args[1]['port'] == 3000
    assert call_args[1]['reload'] is False


def test_apply_env_overrides_loads_file(tmp_path):
    """Ensure env files referenced via --env are loaded."""
    env_file = tmp_path / '.env.dev'
    env_file.write_text('API_TOKEN=abc123\nexport FEATURE_FLAG=true\n# comment\nINVALID_LINE\n')

    with mock.patch.dict(os.environ, {}, clear=True):
        _apply_env_overrides([str(env_file)])
        assert os.environ['API_TOKEN'] == 'abc123'
        assert os.environ['FEATURE_FLAG'] == 'true'
        assert 'INVALID_LINE' not in os.environ


def test_main_build_subcommand(temp_n3_file, tmp_path, capsys):
    """Test main() with build subcommand."""
    output_dir = tmp_path / "build"
    
    main(['build', str(temp_n3_file), '--out', str(output_dir)])
    
    assert output_dir.exists()
    captured = capsys.readouterr()
    assert "Static site generated" in captured.out


def test_main_legacy_invocation(temp_n3_file, tmp_path, capsys):
    """Test main() with legacy invocation (backward compatibility)."""
    output_dir = tmp_path / "build"
    
    main([str(temp_n3_file), '--out', str(output_dir)])
    
    assert output_dir.exists()
    captured = capsys.readouterr()
    assert "legacy invocation" in captured.err
    assert "Static site generated" in captured.out


def test_cmd_train_without_trainer_returns_error(temp_n3_file, capsys, monkeypatch):
    args = mock.Mock()
    args.file = str(temp_n3_file)
    args.model = "demo_model"

    with mock.patch("namel3ss.cli._resolve_model_spec", return_value={"framework": "custom", "metadata": {}}):
        cmd_train(args)

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["error"] == "trainer_not_configured"
def test_cmd_deploy_without_deployer_returns_error(temp_n3_file, capsys, monkeypatch):
    args = mock.Mock()
    args.file = str(temp_n3_file)
    args.model = "demo_model"

    with mock.patch("namel3ss.cli._resolve_model_spec", return_value={"framework": "custom", "version": "v1", "metadata": {}}):
        cmd_deploy(args)

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["error"] == "deployer_not_configured"


def test_main_no_command(capsys):
    """Test main() with no command prints help."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err


@mock.patch('namel3ss.cli.check_uvicorn_available', return_value=True)
def test_main_run_subcommand(mock_check, temp_n3_file, tmp_path, capsys):
    """Test main() with run subcommand."""
    mock_uvicorn = mock.Mock()
    mock_uvicorn.run = mock.Mock()
    backend_dir = tmp_path / "backend"
    
    with mock.patch.dict('sys.modules', {'uvicorn': mock_uvicorn}):
        main(['run', str(temp_n3_file), '--backend-out', str(backend_dir)])
    
    assert backend_dir.exists()
    mock_uvicorn.run.assert_called_once()
    
    captured = capsys.readouterr()
    assert "dev server running" in captured.out
    assert "datasets available" in captured.out
    assert "connectors registered" in captured.out
    assert "insights routed" in captured.out


def test_main_run_natural_language_env_file(tmp_path, monkeypatch):
    """Run command accepts natural language env-file syntax."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'app.n3').write_text(SAMPLE_N3)
    (tmp_path / '.env.dev').write_text('API_TOKEN=english\n')

    with mock.patch('namel3ss.cli.run_dev_server') as mock_run, mock.patch.dict(os.environ, {}, clear=True):
        main(['run', 'app.n3', 'using', '.env.dev', '--backend-out', str(tmp_path / 'backend')])

        mock_run.assert_called_once()
        called_path = mock_run.call_args.args[0]
        assert isinstance(called_path, Path)
        assert called_path.name == 'app.n3'
        assert os.environ.get('API_TOKEN') == 'english'


def test_main_run_natural_language_alias_default_file(tmp_path, monkeypatch):
    """Run command maps aliases like 'locally' to env files and defaults file."""
    monkeypatch.chdir(tmp_path)
    default_file = tmp_path / 'a_app.n3'
    default_file.write_text(SAMPLE_N3)
    (tmp_path / 'z_app.n3').write_text(SAMPLE_N3)
    (tmp_path / '.env.local').write_text('LOCAL_FLAG=on\n')

    with mock.patch('namel3ss.cli.run_dev_server') as mock_run, mock.patch.dict(os.environ, {}, clear=True):
        main(['run', 'locally', '--backend-out', str(tmp_path / 'backend')])

        mock_run.assert_called_once()
        called_path = mock_run.call_args.args[0]
        assert called_path == default_file
        assert os.environ.get('LOCAL_FLAG') == 'on'


def test_main_run_env_file_missing(tmp_path, monkeypatch, capsys):
    """Missing env files referenced via natural language exit gracefully."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'app.n3').write_text(SAMPLE_N3)

    with pytest.raises(SystemExit) as exc, mock.patch('namel3ss.cli.run_dev_server'):
        main(['run', 'using', '.env.missing'])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert 'environment file not found' in captured.err
