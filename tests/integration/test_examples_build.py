"""
Integration tests for example applications.

This module tests that all example applications build successfully
and validates their core functionality.
"""
import os
import subprocess
import tempfile
import pytest
from pathlib import Path


class TestExampleBuilds:
    """Test that all examples build successfully."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def examples_dir(self, project_root):
        """Get the examples directory."""
        return project_root / "examples"
    
    def test_minimal_example_builds(self, examples_dir):
        """Test that the minimal example builds successfully."""
        minimal_dir = examples_dir / "minimal"
        app_file = minimal_dir / "app.ai"
        
        assert app_file.exists(), f"app.ai not found in {minimal_dir}"
        
        result = subprocess.run(
            ["namel3ss", "build", str(app_file)],
            cwd=minimal_dir,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        assert "Build successful" in result.stdout or result.stderr == "", \
            f"Unexpected build output: {result.stdout}"
    
    def test_content_analyzer_example_builds(self, examples_dir):
        """Test that the content analyzer example builds successfully."""
        analyzer_dir = examples_dir / "content-analyzer"
        app_file = analyzer_dir / "app.ai"
        
        assert app_file.exists(), f"app.ai not found in {analyzer_dir}"
        
        result = subprocess.run(
            ["namel3ss", "build", str(app_file)],
            cwd=analyzer_dir,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        assert "Build successful" in result.stdout or result.stderr == "", \
            f"Unexpected build output: {result.stdout}"
    
    def test_research_assistant_example_builds(self, examples_dir):
        """Test that the research assistant example builds successfully."""
        research_dir = examples_dir / "research-assistant"
        app_file = research_dir / "app.ai"
        
        assert app_file.exists(), f"app.ai not found in {research_dir}"
        
        result = subprocess.run(
            ["namel3ss", "build", str(app_file)],
            cwd=research_dir,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        assert "Build successful" in result.stdout or result.stderr == "", \
            f"Unexpected build output: {result.stdout}"
    
    def test_all_examples_build_sequentially(self, examples_dir):
        """Test building all examples in sequence to catch interaction issues."""
        example_dirs = [
            "minimal",
            "content-analyzer", 
            "research-assistant"
        ]
        
        build_results = []
        
        for example_name in example_dirs:
            example_dir = examples_dir / example_name
            app_file = example_dir / "app.ai"
            
            if not app_file.exists():
                pytest.fail(f"app.ai not found in {example_dir}")
            
            result = subprocess.run(
                ["namel3ss", "build", str(app_file)],
                cwd=example_dir,
                capture_output=True,
                text=True
            )
            
            build_results.append({
                "name": example_name,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
        
        # Check all builds succeeded
        failures = [r for r in build_results if r["returncode"] != 0]
        
        if failures:
            failure_details = "\n".join([
                f"  {f['name']}: {f['stderr']}" for f in failures
            ])
            pytest.fail(f"The following examples failed to build:\n{failure_details}")


class TestExampleStructure:
    """Test that examples have proper structure."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def examples_dir(self, project_root):
        """Get the examples directory."""
        return project_root / "examples"
    
    def test_examples_have_app_n3(self, examples_dir):
        """Test that each example has an app.ai file."""
        example_dirs = [
            "minimal",
            "content-analyzer",
            "research-assistant"
        ]
        
        for example_name in example_dirs:
            example_dir = examples_dir / example_name
            app_file = example_dir / "app.ai"
            
            assert example_dir.is_dir(), f"Example directory {example_dir} not found"
            assert app_file.exists(), f"app.ai not found in {example_dir}"
    
    def test_examples_have_readme(self, examples_dir):
        """Test that each example has a README.md file."""
        example_dirs = [
            "minimal",
            "content-analyzer", 
            "research-assistant"
        ]
        
        for example_name in example_dirs:
            example_dir = examples_dir / example_name
            readme_file = example_dir / "README.md"
            
            assert readme_file.exists(), f"README.md not found in {example_dir}"
            
            # Check README has basic content
            readme_content = readme_file.read_text()
            assert example_name.replace("-", " ").title() in readme_content, \
                f"README.md in {example_dir} should mention the example name"


class TestFixtureStructure:
    """Test that test fixtures are properly organized."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def tests_dir(self, project_root):
        """Get the tests directory."""
        return project_root / "tests"
    
    def test_fixture_directories_exist(self, tests_dir):
        """Test that fixture directories are properly structured."""
        expected_dirs = [
            "unit/fixtures/agents",
            "unit/fixtures/prompts", 
            "unit/fixtures/llms",
            "unit/fixtures/syntax",
            "integration/fixtures/templates"
        ]
        
        for fixture_path in expected_dirs:
            fixture_dir = tests_dir / fixture_path
            assert fixture_dir.is_dir(), f"Fixture directory {fixture_dir} not found"
    
    def test_syntax_fixtures_exist(self, tests_dir):
        """Test that syntax test fixtures exist."""
        syntax_fixtures_dir = tests_dir / "unit/fixtures/syntax"
        
        # Should have the LSP test files
        expected_files = [
            "dashboard.ai",
            "metrics.ai", 
            "syntax_error.ai",
            "type_error.ai"
        ]
        
        for filename in expected_files:
            fixture_file = syntax_fixtures_dir / filename
            assert fixture_file.exists(), f"Syntax fixture {fixture_file} not found"
    
    def test_integration_templates_exist(self, tests_dir):
        """Test that integration test templates exist."""
        templates_base_dir = tests_dir / "integration/fixtures/templates"
        
        expected_templates = [
            templates_base_dir / "minimal/app.ai",
            templates_base_dir / "agent/app.ai"
        ]
        
        for template_file in expected_templates:
            assert template_file.exists(), f"Integration template {template_file} not found"


class TestFixtureValidity:
    """Test that fixture files are valid N3 syntax."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def tests_dir(self, project_root):
        """Get the tests directory."""
        return project_root / "tests"
    
    def test_integration_templates_build(self, tests_dir):
        """Test that integration templates build successfully."""
        templates_base_dir = tests_dir / "integration/fixtures/templates"
        
        template_apps = [
            templates_base_dir / "minimal/app.ai",
            templates_base_dir / "agent/app.ai"
        ]
        
        for template_file in template_apps:
            result = subprocess.run(
                ["namel3ss", "build", str(template_file)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, \
                f"Integration template {template_file} failed to build: {result.stderr}"