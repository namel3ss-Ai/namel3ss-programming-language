"""
Language-level integration tests for Namel3ss.

Tests validate that .n3 source files:
- Parse correctly through the grammar
- Resolve symbols and types without errors
- Generate backend and frontend code successfully
- All examples in the repository remain compilable

These tests ensure the end-to-end compilation pipeline works.
"""

from pathlib import Path
from typing import List

import pytest

from namel3ss.ast import App, Module, Program
from namel3ss.codegen import generate_backend, generate_site
from namel3ss.loader import load_program, extract_single_app
from namel3ss.parser import Parser, N3SyntaxError
from namel3ss.resolver import resolve_program, ModuleResolutionError


# ============================================================================
# Parsing Tests
# ============================================================================

def test_minimal_app_parses():
    """Test that minimal valid app parses."""
    source = 'app "Minimal".'
    
    app = Parser(source).parse_app()
    
    assert app is not None
    assert app.name == "Minimal"


def test_app_with_page_parses():
    """Test that app with page parses correctly."""
    source = '''
app "TestApp".

page "Home" at "/":
  show text "Welcome"
'''
    
    app = Parser(source).parse_app()
    
    assert len(app.pages) == 1
    assert app.pages[0].name == "Home"
    assert app.pages[0].path == "/"


def test_app_with_dataset_parses():
    """Test that app with dataset parses correctly."""
    source = '''
app "DataApp" connects to postgres "DB".

dataset "users" from table users:
  filter by: status == "active"
'''
    
    app = Parser(source).parse_app()
    
    assert len(app.datasets) == 1
    assert app.datasets[0].name == "users"


def test_app_with_prompt_parses():
    """Test that app with structured prompt parses correctly."""
    pytest.skip("AI grammar (prompts) not yet integrated into main parser")
    source = '''
app "AIApp".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
}

prompt "classify" {
  args: {
    text: string
  }
  output_schema: {
    category: enum["a", "b", "c"]
  }
  model: "gpt4"
  template: "Classify: {text}"
}
'''
    
    module = Parser(source).parse()
    app = module.apps[0]
    
    assert len(app.prompts) == 1
    assert app.prompts[0].name == "classify"
    assert app.prompts[0].output_schema is not None


def test_app_with_control_flow_parses():
    """Test that control flow syntax parses correctly."""
    source = '''
app "ControlApp".

dataset "items" from table items:
  add column id = 1

page "Test" at "/test":
  if user.role == "admin":
    show text "Admin"
  else:
    show text "User"
  
  for item in dataset items:
    show text "{item.name}"
'''
    
    app = Parser(source).parse_app()
    page = app.pages[0]
    
    # Page should have components (control flow creates components)
    assert len(page.components) > 0


def test_app_with_memory_parses():
    """Test that memory blocks parse correctly."""
    pytest.skip("Memory blocks not yet in main grammar")
    source = '''
app "MemoryApp".

memory "history" {
  scope: "user"
  kind: "list"
  max_items: 100
}
'''
    
    module = Parser(source).parse()
    app = module.apps[0]
    
    assert len(app.memories) == 1
    assert app.memories[0].name == "history"


def test_app_with_agent_parses():
    """Test that agent blocks parse correctly."""
    pytest.skip("Agent blocks not yet in main grammar")
    source = '''
app "AgentApp".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
}

agent researcher {
  llm: gpt4
  goal: "Research topics"
}
'''
    
    module = Parser(source).parse()
    app = module.apps[0]
    
    assert len(app.agents) == 1
    assert app.agents[0].name == "researcher"


def test_app_with_chain_parses():
    """Test that chain definitions parse correctly."""
    pytest.skip("Chain blocks not yet in main grammar")
    source = '''
app "ChainApp".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
}

prompt "step1" {
  model: "gpt4"
  template: "Step 1"
}

define chain "test_chain" {
  steps:
    - step "s1" {
        kind: "prompt"
        target: "step1"
      }
}
'''
    
    module = Parser(source).parse()
    app = module.apps[0]
    
    assert len(app.chains) == 1
    assert app.chains[0].name == "test_chain"


# ============================================================================
# Syntax Error Tests
# ============================================================================

def test_invalid_syntax_raises_clear_error():
    """Test that invalid syntax produces clear error message."""
    source = 'app "Test" {'  # Missing closing brace
    
    with pytest.raises(N3SyntaxError) as exc_info:
        Parser(source).parse()
    
    error = str(exc_info.value)
    assert "syntax" in error.lower() or "unexpected" in error.lower()


def test_missing_quote_raises_clear_error():
    """Test that missing quote produces helpful error."""
    source = 'app "Test.'  # Missing closing quote
    
    with pytest.raises(N3SyntaxError) as exc_info:
        Parser(source).parse()
    
    # Error should mention the issue
    error = str(exc_info.value)
    assert error  # Non-empty error message


# ============================================================================
# Resolution Tests
# ============================================================================

def test_simple_program_resolves():
    """Test that simple program resolves without errors."""
    source = '''
app "ResolveTest".

page "Home" at "/":
  show text "Test"
'''
    
    program = load_program(Path("test.n3"))
    # Would need actual file for load_program, so use parser directly
    module = Parser(source).parse()
    program = Program(modules=[module])
    
    resolved = resolve_program(program)
    
    assert resolved.app is not None
    assert resolved.app.name == "ResolveTest"


def test_undefined_dataset_raises_resolution_error():
    """Test that referencing undefined dataset produces clear error."""
    source = '''
app "Test".

page "Home" at "/":
  show table "Data" from dataset undefined_dataset
'''
    
    module = Parser(source).parse()
    program = Program(modules=[module])
    
    with pytest.raises(ModuleResolutionError) as exc_info:
        resolve_program(program)
    
    error = str(exc_info.value)
    assert "undefined_dataset" in error


def test_undefined_prompt_raises_resolution_error():
    """Test that referencing undefined prompt produces clear error."""
    source = '''
app "Test".

define chain "test" {
  steps:
    - step "s1" {
        kind: "prompt"
        target: "nonexistent_prompt"
      }
}
'''
    
    module = Parser(source).parse()
    program = Program(modules=[module])
    
    with pytest.raises(ModuleResolutionError) as exc_info:
        resolve_program(program)
    
    error = str(exc_info.value)
    assert "nonexistent_prompt" in error or "not found" in error.lower()


# ============================================================================
# Code Generation Tests
# ============================================================================

def test_backend_generation_succeeds(tmp_path: Path):
    """Test that backend code generation completes without errors."""
    source = '''
app "GenTest".

page "Home" at "/":
  show text "Test"
'''
    
    app = Parser(source).parse_app()
    backend_dir = tmp_path / "backend"
    
    generate_backend(app, backend_dir)
    
    # Check that key files were created
    assert (backend_dir / "main.py").exists()
    assert (backend_dir / "generated").exists()
    assert (backend_dir / "generated" / "registries.py").exists()


def test_frontend_generation_succeeds(tmp_path: Path):
    """Test that frontend code generation completes without errors."""
    source = '''
app "GenTest".

page "Home" at "/":
  show text "Test"
'''
    
    app = Parser(source).parse_app()
    frontend_dir = tmp_path / "frontend"
    
    generate_site(app, frontend_dir)
    
    # Check that key files were created
    assert (frontend_dir / "index.html").exists()


def test_app_with_all_features_generates(tmp_path: Path):
    """Test that app using multiple features generates successfully."""
    source = '''
app "CompleteApp" connects to postgres "DB".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
}

dataset "users" from table users:
  filter by: active == true

memory "history" {
  scope: "session"
  kind: "list"
}

prompt "classify" {
  args: {
    text: string
  }
  output_schema: {
    category: enum["a", "b"]
  }
  model: "gpt4"
  template: "Classify: {text}"
}

page "Home" at "/":
  if user.authenticated:
    show text "Welcome"
  else:
    show text "Please login"
  
  for user in dataset users:
    show text "{user.name}"
'''
    
    app = Parser(source).parse_app()
    backend_dir = tmp_path / "backend"
    
    # Should complete without errors
    generate_backend(app, backend_dir)
    
    assert (backend_dir / "main.py").exists()


# ============================================================================
# Example File Tests
# ============================================================================

def _find_example_files() -> List[Path]:
    """Find all .n3 example files."""
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        return []
    return list(examples_dir.glob("*.n3"))


@pytest.mark.parametrize("example_path", _find_example_files())
def test_example_file_compiles(example_path: Path, tmp_path: Path):
    """Test that each example file compiles successfully."""
    try:
        # Load and parse
        program = load_program(str(example_path))
        
        # Extract app
        app = extract_single_app(program)
        
        # Generate backend (smoke test)
        backend_dir = tmp_path / f"backend_{example_path.stem}"
        generate_backend(app, backend_dir)
        
        # Verify key files exist
        assert (backend_dir / "main.py").exists()
        
    except Exception as e:
        pytest.fail(f"Failed to compile {example_path.name}: {type(e).__name__}: {e}")


def test_demo_app_compiles():
    """Test that demo_app.n3 compiles successfully."""
    demo_path = Path(__file__).parent.parent / "demo_app.n3"
    if not demo_path.exists():
        pytest.skip("demo_app.n3 not found")
    
    program = load_program(str(demo_path))
    app = extract_single_app(program)
    
    assert app is not None
    assert app.name  # Should have a name


# ============================================================================
# Module Loading Tests
# ============================================================================

def test_load_program_from_file(tmp_path: Path):
    """Test loading program from .n3 file."""
    test_file = tmp_path / "test.n3"
    test_file.write_text('app "FileTest".\npage "Home" at "/": show text "Test"')
    
    program = load_program(str(test_file))
    
    assert len(program.modules) == 1
    assert len(program.modules[0].apps) == 1


def test_load_program_from_directory(tmp_path: Path):
    """Test loading multiple .n3 files from directory."""
    (tmp_path / "app1.n3").write_text('app "App1".')
    (tmp_path / "app2.n3").write_text('app "App2".')
    
    program = load_program(str(tmp_path))
    
    # Should load multiple modules
    assert len(program.modules) >= 2


def test_load_nonexistent_file_raises_error():
    """Test that loading nonexistent file raises clear error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_program("nonexistent.n3")
    
    error = str(exc_info.value)
    assert "nonexistent.n3" in error or "not found" in error.lower()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline_simple_app(tmp_path: Path):
    """Test complete pipeline: parse -> resolve -> generate."""
    source = '''
app "PipelineTest".

dataset "data" from table items:
  add column id = 1

page "Home" at "/":
  show text "Welcome"
  show table "Items" from dataset data
'''
    
    # Parse
    app = Parser(source).parse_app()
    assert app is not None
    
    # Resolve
    module = Parser(source).parse()
    program = Program(modules=[module])
    resolved = resolve_program(program)
    assert resolved.app is not None
    
    # Generate
    backend_dir = tmp_path / "backend"
    generate_backend(resolved.app, backend_dir)
    assert (backend_dir / "main.py").exists()


def test_full_pipeline_ai_features(tmp_path: Path):
    """Test pipeline with AI features."""
    pytest.skip("AI features not yet in main grammar - tested separately")
    source = '''
app "AITest".

llm gpt4 {
  provider: "openai"
  model: "gpt-4"
}

memory "context" {
  scope: "session"
  kind: "list"
  max_items: 50
}

prompt "analyze" {
  args: {
    text: string
  }
  output_schema: {
    sentiment: enum["positive", "negative", "neutral"],
    confidence: float
  }
  model: "gpt4"
  template: "Analyze sentiment of: {text}\n\nContext: {memory.context}"
}

define chain "analysis" {
  steps:
    - step "analyze_step" {
        kind: "prompt"
        target: "analyze"
      }
    - step "save_result" {
        kind: "memory_write"
        target: "context"
      }
}

page "Home" at "/":
  show text "AI Analysis App"
'''
    
    # Parse
    module = Parser(source).parse()
    app = module.apps[0]
    
    # Verify all AI features were parsed
    assert len(app.llms) == 1
    assert len(app.memories) == 1
    assert len(app.prompts) == 1
    assert len(app.chains) == 1
    
    # Resolve
    program = Program(modules=[module])
    resolved = resolve_program(program)
    
    # Generate
    backend_dir = tmp_path / "backend"
    generate_backend(resolved.app, backend_dir)
    assert (backend_dir / "main.py").exists()


def test_error_recovery_continues_after_resolution_error():
    """Test that parser can continue after encountering error."""
    source = '''
app "ErrorTest".

page "Valid" at "/":
  show text "This is valid"

page "Invalid" at "/invalid":
  show table "Data" from dataset nonexistent
'''
    
    module = Parser(source).parse()
    program = Program(modules=[module])
    
    # Resolution should fail on undefined dataset
    with pytest.raises(ModuleResolutionError):
        resolve_program(program)
    
    # But parsing succeeded - module has content
    assert module.has_explicit_app
    assert len(module.body) > 0


# ============================================================================
# Regression Tests
# ============================================================================

def test_regression_empty_app_handling():
    """Regression: ensure empty apps are handled gracefully."""
    source = 'app "Empty".'
    
    app = Parser(source).parse_app()
    module = Parser(source).parse()
    program = Program(modules=[module])
    resolved = resolve_program(program)
    
    assert resolved.app.name == "Empty"
    assert len(resolved.app.pages) == 0


def test_regression_special_characters_in_strings():
    """Regression: ensure special characters in strings are handled."""
    source = r'''
app "SpecialChars".

page "Test" at "/test":
  show text "Special: \n \t \" \\"
'''
    
    app = Parser(source).parse_app()
    assert app is not None


def test_regression_nested_control_flow():
    """Regression: ensure nested if/for statements work."""
    source = '''
app "NestedTest".

dataset "items" from table items:
  add column id = 1

page "Test" at "/":
  for item in dataset items:
    if item.active:
      show text "{item.name}"
    else:
      show text "Inactive"
'''
    
    app = Parser(source).parse_app()
    assert app is not None
