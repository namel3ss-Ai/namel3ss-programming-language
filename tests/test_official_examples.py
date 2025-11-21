"""
Test suite for official N3 examples.

Ensures all official .n3 examples build successfully and deterministically.
This is the core stability test for the compiler-to-backend pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
import json
import hashlib

from namel3ss.parser import Parser
from namel3ss.codegen.backend.core import generate_backend
from namel3ss.codegen.frontend import generate_site


# Official examples that MUST build successfully
OFFICIAL_EXAMPLES = [
    "demo_app.n3",
    "examples/simple_functional.n3",
    "examples/provider_demo.n3",
    "examples/rag_demo.n3",
    "examples/symbolic_demo.n3",
    "examples/template_examples.n3",
    "examples/safety_policies.n3",
    "examples/advanced_providers.n3",
    "examples/memory_chat_demo.n3",
    "examples/multimodal_rag.n3",
]


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def get_official_examples() -> List[Path]:
    """Get list of official example .n3 files."""
    repo_root = get_repo_root()
    examples = []
    
    for example_path in OFFICIAL_EXAMPLES:
        full_path = repo_root / example_path
        if full_path.exists():
            examples.append(full_path)
        else:
            pytest.skip(f"Example not found: {example_path}")
    
    return examples


def compute_directory_hash(directory: Path) -> str:
    """
    Compute deterministic hash of directory contents.
    
    This ensures builds are deterministic - same input produces same output.
    """
    hasher = hashlib.sha256()
    
    # Get all files in sorted order for determinism
    files = sorted(directory.rglob("*"))
    
    for file_path in files:
        if file_path.is_file():
            # Hash relative path
            rel_path = file_path.relative_to(directory)
            hasher.update(str(rel_path).encode('utf-8'))
            
            # Hash file contents
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    
    return hasher.hexdigest()


class TestOfficialExamples:
    """Test that all official examples build successfully."""
    
    @pytest.mark.parametrize("example_path", get_official_examples())
    def test_example_parses(self, example_path: Path):
        """Test that example parses without errors."""
        with open(example_path, 'r') as f:
            content = f.read()
        
        # Parser requires source as argument
        parser = Parser(content)
        
        # Should not raise
        ast = parser.parse_app()
        
        # Verify AST is not empty
        assert ast is not None
        assert hasattr(ast, 'name')
    
    @pytest.mark.parametrize("example_path", get_official_examples())
    def test_example_builds_backend(self, example_path: Path, tmp_path: Path):
        """Test that example builds backend successfully."""
        with open(example_path, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Generate backend
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        
        # Should not raise
        generate_backend(
            ast,
            str(backend_dir),
            embed_insights=False,
            enable_realtime=False,
        )
        
        # Verify expected files exist
        assert (backend_dir / "main.py").exists()
        assert (backend_dir / "runtime.py").exists()
        
        # Verify main.py has FastAPI app
        main_content = (backend_dir / "main.py").read_text()
        assert "from fastapi import FastAPI" in main_content
        assert "app = FastAPI" in main_content
    
    @pytest.mark.parametrize("example_path", get_official_examples())
    def test_example_builds_frontend(self, example_path: Path, tmp_path: Path):
        """Test that example builds frontend successfully."""
        with open(example_path, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Generate frontend
        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        
        # Should not raise
        generate_site(
            ast,
            str(frontend_dir),
            enable_realtime=False,
            target="static",
        )
        
        # Verify expected files exist
        assert (frontend_dir / "index.html").exists()
    
    @pytest.mark.parametrize("example_path", get_official_examples())
    def test_example_build_determinism(self, example_path: Path, tmp_path: Path):
        """
        Test that builds are deterministic.
        
        Building the same .n3 file twice should produce identical output.
        This is critical for reproducible builds and CI/CD.
        """
        with open(example_path, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # First build
        backend_dir1 = tmp_path / "backend1"
        backend_dir1.mkdir()
        generate_backend(ast, str(backend_dir1), embed_insights=False)
        hash1 = compute_directory_hash(backend_dir1)
        
        # Second build
        backend_dir2 = tmp_path / "backend2"
        backend_dir2.mkdir()
        generate_backend(ast, str(backend_dir2), embed_insights=False)
        hash2 = compute_directory_hash(backend_dir2)
        
        # Hashes should be identical
        assert hash1 == hash2, f"Build is non-deterministic for {example_path.name}"


class TestBuildStability:
    """Test build pipeline stability and error handling."""
    
    def test_invalid_syntax_fails_gracefully(self, tmp_path: Path):
        """Test that invalid syntax produces clear error."""
        invalid_n3 = tmp_path / "invalid.n3"
        invalid_n3.write_text("this is not valid N3 syntax !!!")
        
        parser = Parser()
        
        # Should raise parse error
        with pytest.raises(Exception) as exc_info:
            with open(invalid_n3, 'r') as f:
                parser.parse(f.read())
        
        # Error should mention syntax or parse
        error_msg = str(exc_info.value).lower()
        assert "syntax" in error_msg or "parse" in error_msg or "unexpected" in error_msg
    
    def test_empty_app_builds(self, tmp_path: Path):
        """Test that minimal empty app builds successfully."""
        minimal_n3 = tmp_path / "minimal.n3"
        minimal_n3.write_text("""
app "minimal_app" {
  description: "A minimal test app"
}
""")
        
        with open(minimal_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Generate backend
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        
        generate_backend(ast, str(backend_dir))
        
        # Verify files exist
        assert (backend_dir / "main.py").exists()
        assert (backend_dir / "runtime.py").exists()
    
    def test_backend_output_structure(self, tmp_path: Path):
        """Test that backend output has consistent structure."""
        test_n3 = tmp_path / "test.n3"
        test_n3.write_text("""
app "test_app" {
  description: "Test app"
}

prompt "test_prompt" {
  template: "Hello {{name}}"
}
""")
        
        with open(test_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        
        generate_backend(ast, str(backend_dir))
        
        # Check expected structure
        assert (backend_dir / "main.py").exists()
        assert (backend_dir / "runtime.py").exists()
        
        # Verify main.py structure
        main_py = (backend_dir / "main.py").read_text()
        assert "def create_app()" in main_py or "app = FastAPI" in main_py
        
        # Verify runtime.py has prompt implementation
        runtime_py = (backend_dir / "runtime.py").read_text()
        assert "test_prompt" in runtime_py.lower() or "prompt" in runtime_py.lower()


class TestBuildIntegration:
    """Integration tests for full build pipeline."""
    
    def test_full_build_pipeline(self, tmp_path: Path):
        """Test complete parse → backend → frontend pipeline."""
        test_n3 = tmp_path / "full_test.n3"
        test_n3.write_text("""
app "full_test" {
  description: "Full pipeline test"
  version: "1.0.0"
}

page "home" {
  title: "Home"
  layout: "default"
}

prompt "greeting" {
  template: "Hello {{name}}!"
}
""")
        
        with open(test_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Generate backend
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        generate_backend(ast, str(backend_dir))
        
        # Generate frontend
        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        generate_site(ast, str(frontend_dir), target="static")
        
        # Verify both generated successfully
        assert (backend_dir / "main.py").exists()
        assert (frontend_dir / "index.html").exists()
    
    def test_build_with_connectors(self, tmp_path: Path):
        """Test build with database/redis connectors."""
        test_n3 = tmp_path / "connectors.n3"
        test_n3.write_text("""
app "connector_test" {
  description: "Test connectors"
}

connectors {
  database: {
    provider: "postgresql"
    connection_string: env("DATABASE_URL")
  }
}

prompt "test" {
  template: "Test"
}
""")
        
        with open(test_n3, 'r') as f:
            content = f.read()
        
        # Should parse even with connectors
        parser = Parser(content)
        ast = parser.parse_app()
        
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        
        # Should generate backend with connector config
        generate_backend(ast, str(backend_dir))
        
        assert (backend_dir / "main.py").exists()


class TestDeterminism:
    """Test suite specifically for deterministic behavior."""
    
    def test_file_generation_order(self, tmp_path: Path):
        """Test that generated files appear in consistent order."""
        test_n3 = tmp_path / "order_test.n3"
        test_n3.write_text("""
app "order_test" {
  description: "Test file order"
}

prompt "prompt_a" { template: "A" }
prompt "prompt_b" { template: "B" }
prompt "prompt_c" { template: "C" }
""")
        
        with open(test_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Build twice
        builds = []
        for i in range(2):
            backend_dir = tmp_path / f"backend_{i}"
            backend_dir.mkdir()
            generate_backend(ast, str(backend_dir))
            
            # Get sorted file list
            files = sorted([f.name for f in backend_dir.rglob("*") if f.is_file()])
            builds.append(files)
        
        # Both builds should have same file list in same order
        assert builds[0] == builds[1]
    
    def test_content_determinism(self, tmp_path: Path):
        """Test that generated file content is deterministic."""
        test_n3 = tmp_path / "content_test.n3"
        test_n3.write_text("""
app "content_test" {}
prompt "test" { template: "Test {{var}}" }
""")
        
        with open(test_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        # Build twice and compare main.py content
        contents = []
        for i in range(2):
            backend_dir = tmp_path / f"backend_{i}"
            backend_dir.mkdir()
            generate_backend(ast, str(backend_dir))
            
            main_content = (backend_dir / "main.py").read_text()
            contents.append(main_content)
        
        # Content should be identical
        assert contents[0] == contents[1]


@pytest.mark.slow
class TestLargeExamples:
    """Tests for larger, more complex examples (marked slow)."""
    
    def test_complex_app_builds(self, tmp_path: Path):
        """Test that complex apps with multiple features build."""
        complex_n3 = tmp_path / "complex.n3"
        complex_n3.write_text("""
app "complex_app" {
  description: "Complex test app"
  version: "1.0.0"
}

page "home" { title: "Home" }
page "about" { title: "About" }

prompt "greeting" { template: "Hello {{name}}" }
prompt "farewell" { template: "Goodbye {{name}}" }

chain "process" {
  inputs: { data: any }
  outputs: { result: any }
  steps: [
    { call: "greeting", inputs: { name: "{{data.name}}" } }
  ]
}
""")
        
        with open(complex_n3, 'r') as f:
            content = f.read()
        
        parser = Parser(content)
        ast = parser.parse_app()
        
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        
        generate_backend(ast, str(backend_dir))
        
        # Verify structure
        assert (backend_dir / "main.py").exists()
        assert (backend_dir / "runtime.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
