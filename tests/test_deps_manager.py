"""
Tests for namel3ss.deps.manager module.

Tests the high-level DependencyManager API.
"""

import pytest
import json
from pathlib import Path
from namel3ss.deps.manager import DependencyManager


class TestDependencyManager:
    """Test DependencyManager class"""
    
    def test_manager_creation(self):
        """Test creating a DependencyManager"""
        manager = DependencyManager()
        assert manager is not None
    
    def test_manager_verbose_mode(self):
        """Test creating manager in verbose mode"""
        manager = DependencyManager(verbose=True)
        assert manager.verbose is True


class TestListAvailableFeatures:
    """Test list_available_features method"""
    
    def test_list_available_features(self):
        """Test listing available features"""
        manager = DependencyManager()
        features = manager.list_available_features()
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check structure
        for feature_id, info in features.items():
            assert 'category' in info
            assert 'description' in info
            assert 'python_packages' in info
            assert 'npm_packages' in info
    
    def test_list_features_includes_core(self):
        """Test that core features are listed"""
        manager = DependencyManager()
        features = manager.list_available_features()
        
        assert 'core' in features
        assert 'openai' in features
        assert 'postgres' in features


class TestPreviewDependencies:
    """Test preview_dependencies method"""
    
    def test_preview_empty_project(self):
        """Test previewing dependencies for empty project"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DependencyManager()
            result = manager.preview_dependencies(tmpdir)
            
            assert 'features' in result
            assert 'added_python' in result
            assert 'added_npm' in result
            assert 'warnings' in result
    
    def test_preview_with_ai_file(self):
        """Test previewing dependencies with .ai file"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test .ai file
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.preview_dependencies(tmpdir)
            
            assert 'openai' in result['features']
            assert len(result['added_python']) > 0


class TestSyncFromFile:
    """Test sync_from_file method"""
    
    def test_sync_from_file_creates_deps(self):
        """Test syncing from single file creates dependency files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test .ai file
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_from_file(ai_file)
            
            # Check that files were created
            req_file = Path(tmpdir) / "requirements.txt"
            pkg_file = Path(tmpdir) / "package.json"
            
            assert req_file.exists()
            assert pkg_file.exists()
            
            # Check content
            req_content = req_file.read_text(encoding='utf-8')
            assert "openai" in req_content
    
    def test_sync_from_file_preview_mode(self):
        """Test syncing in preview mode doesn't write files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_from_file(ai_file, preview=True)
            
            # Files should NOT be created
            req_file = Path(tmpdir) / "requirements.txt"
            pkg_file = Path(tmpdir) / "package.json"
            
            assert not req_file.exists()
            assert not pkg_file.exists()
            
            # But result should show what would be added
            assert 'openai' in result['features']


class TestSyncProject:
    """Test sync_project method"""
    
    def test_sync_project_standard_layout(self):
        """Test syncing project with backend/frontend layout"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create standard layout
            backend_dir = Path(tmpdir) / "backend"
            frontend_dir = Path(tmpdir) / "frontend"
            backend_dir.mkdir()
            frontend_dir.mkdir()
            
            # Create app.ai
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_project(tmpdir)
            
            # Check that files were created in correct locations
            assert (backend_dir / "requirements.txt").exists()
            assert (frontend_dir / "package.json").exists()
    
    def test_sync_project_root_layout(self):
        """Test syncing project with root-level dep files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create app.ai
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_project(tmpdir)
            
            # Check that files were created at root
            assert (Path(tmpdir) / "requirements.txt").exists()
            assert (Path(tmpdir) / "package.json").exists()
    
    def test_sync_project_multiple_ai_files(self):
        """Test syncing project with multiple .ai files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple .ai files
            ai_file1 = Path(tmpdir) / "app1.ai"
            ai_file1.write_text("""
app "App1" {
    description: "Test 1"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            ai_file2 = Path(tmpdir) / "app2.ai"
            ai_file2.write_text("""
app "App2" {
    description: "Test 2"
}

llm claude {
    provider: "anthropic"
    model: "claude-3-opus"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_project(tmpdir)
            
            # Should detect features from both files
            assert 'openai' in result['features']
            assert 'anthropic' in result['features']
    
    def test_sync_project_no_ai_files(self):
        """Test syncing project with no .ai files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DependencyManager()
            result = manager.sync_project(tmpdir)
            
            # Should have warnings
            assert len(result['warnings']) > 0


class TestNonDestructiveUpdates:
    """Test non-destructive update behavior in manager"""
    
    def test_preserves_existing_python_deps(self):
        """Test that existing Python dependencies are preserved"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backend dir with existing requirements
            backend_dir = Path(tmpdir) / "backend"
            backend_dir.mkdir()
            
            req_file = backend_dir / "requirements.txt"
            req_file.write_text("custom-package==1.0.0\n", encoding='utf-8')
            
            # Create app.ai
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            manager.sync_project(tmpdir)
            
            # Check that custom package is preserved
            content = req_file.read_text(encoding='utf-8')
            assert "custom-package==1.0.0" in content
            assert "openai" in content
    
    def test_preserves_existing_npm_deps(self):
        """Test that existing NPM dependencies are preserved"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create frontend dir with existing package.json
            frontend_dir = Path(tmpdir) / "frontend"
            frontend_dir.mkdir()
            
            pkg_file = frontend_dir / "package.json"
            existing_pkg = {
                "name": "my-app",
                "version": "1.0.0",
                "dependencies": {
                    "custom-lib": "^1.0.0"
                },
                "devDependencies": {}
            }
            pkg_file.write_text(json.dumps(existing_pkg, indent=2), encoding='utf-8')
            
            # Create app.ai (doesn't matter what's in it)
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            manager.sync_project(tmpdir)
            
            # Check that custom package is preserved
            content = pkg_file.read_text(encoding='utf-8')
            pkg_json = json.loads(content)
            assert "custom-lib" in pkg_json["dependencies"]


class TestResultFormat:
    """Test result dictionary format"""
    
    def test_result_has_required_keys(self):
        """Test that result has all required keys"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_from_file(ai_file)
            
            assert 'features' in result
            assert 'added_python' in result
            assert 'added_npm' in result
            assert 'warnings' in result
            
            assert isinstance(result['features'], set)
            assert isinstance(result['added_python'], list)
            assert isinstance(result['added_npm'], list)
            assert isinstance(result['warnings'], list)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_sync_nonexistent_file(self):
        """Test syncing from non-existent file"""
        manager = DependencyManager()
        result = manager.sync_from_file("nonexistent.ai")
        
        # Should have warnings
        assert len(result['warnings']) > 0
    
    def test_sync_invalid_ai_file(self):
        """Test syncing from invalid .ai file"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("this is not valid namel3ss {", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_from_file(ai_file)
            
            # Should complete with warnings
            assert 'warnings' in result
            assert len(result['warnings']) > 0
    
    def test_sync_project_deep_nesting(self):
        """Test syncing project with deeply nested .ai files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory
            nested_dir = Path(tmpdir) / "level1" / "level2" / "level3"
            nested_dir.mkdir(parents=True)
            
            ai_file = nested_dir / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager()
            result = manager.sync_project(tmpdir)
            
            # Should find and process the nested file
            assert 'openai' in result['features']


class TestVerboseOutput:
    """Test verbose output mode"""
    
    def test_verbose_mode_enabled(self):
        """Test that verbose mode doesn't break functionality"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ai_file = Path(tmpdir) / "app.ai"
            ai_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            manager = DependencyManager(verbose=True)
            result = manager.sync_from_file(ai_file)
            
            # Should work the same as non-verbose
            assert 'openai' in result['features']
            assert len(result['added_python']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
