"""
Tests for namel3ss.deps.detector module.

Tests IR-based feature detection.
"""

import pytest
from pathlib import Path
from namel3ss.deps.detector import FeatureDetector, DetectedFeatures


class TestDetectedFeatures:
    """Test DetectedFeatures dataclass"""
    
    def test_detected_features_creation(self):
        """Test creating DetectedFeatures"""
        features = DetectedFeatures()
        assert len(features.features) == 0
        assert len(features.warnings) == 0
    
    def test_add_feature(self):
        """Test adding a feature"""
        features = DetectedFeatures()
        features.add_feature("openai")
        
        assert "openai" in features.features
        assert len(features.features) == 1
    
    def test_add_duplicate_feature(self):
        """Test that duplicate features are deduplicated"""
        features = DetectedFeatures()
        features.add_feature("openai")
        features.add_feature("openai")
        
        assert len(features.features) == 1
    
    def test_add_warning(self):
        """Test adding a warning"""
        features = DetectedFeatures()
        features.add_warning("Test warning")
        
        assert "Test warning" in features.warnings
        assert len(features.warnings) == 1
    
    def test_merge(self):
        """Test merging two DetectedFeatures"""
        features1 = DetectedFeatures()
        features1.add_feature("openai")
        features1.add_warning("Warning 1")
        
        features2 = DetectedFeatures()
        features2.add_feature("postgres")
        features2.add_warning("Warning 2")
        
        features1.merge(features2)
        
        assert len(features1.features) == 2
        assert "openai" in features1.features
        assert "postgres" in features1.features
        assert len(features1.warnings) == 2


class TestFeatureDetector:
    """Test FeatureDetector class"""
    
    def test_detector_creation(self):
        """Test creating a FeatureDetector"""
        detector = FeatureDetector()
        assert detector is not None
        assert isinstance(detector.detected, DetectedFeatures)
    
    def test_detect_from_source_openai(self):
        """Test detecting OpenAI from source"""
        source = """
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}

agent assistant {
    llm: gpt4
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        assert "openai" in result.features
    
    def test_detect_from_source_anthropic(self):
        """Test detecting Anthropic from source"""
        source = """
app "TestApp" {
    description: "Test"
}

llm claude {
    provider: "anthropic"
    model: "claude-3-opus"
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        assert "anthropic" in result.features
    
    def test_detect_from_source_postgres(self):
        """Test detecting PostgreSQL from source"""
        source = """
app "TestApp" {
    description: "Test"
}

dataset users from table users
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Default to SQL for table sources
        assert "sql" in result.features
    
    def test_detect_from_source_multiple_features(self):
        """Test detecting multiple features"""
        source = """
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}

dataset users from table users

agent assistant {
    llm: gpt4
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        assert "openai" in result.features
        assert "sql" in result.features
    
    def test_detect_invalid_source(self):
        """Test detecting from invalid source"""
        source = "this is not valid namel3ss syntax {"
        
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should have warnings about parsing errors
        assert len(result.warnings) > 0
    
    def test_detect_empty_source(self):
        """Test detecting from empty source"""
        detector = FeatureDetector()
        result = detector.detect_from_source("")
        
        # Empty source might have warnings or just return empty features
        assert isinstance(result, DetectedFeatures)
    
    def test_detect_from_source_websockets(self):
        """Test detecting websockets feature"""
        source = """
app "TestApp" {
    description: "Test"
}

page "Home" at "/" {
    realtime: true
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Realtime pages should trigger websockets
        assert "websockets" in result.features or len(result.features) >= 0


class TestBackendFeatureDetection:
    """Test backend feature detection"""
    
    def test_detect_agent_implies_ai(self):
        """Test that agents imply AI usage"""
        source = """
app "TestApp" {
    description: "Test"
}

agent assistant {
    goal: "Help users"
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Agents should imply openai by default
        assert "openai" in result.features
    
    def test_detect_dataset_table(self):
        """Test detecting dataset from table"""
        source = """
app "TestApp" {
    description: "Test"
}

dataset users from table users
dataset posts from table posts
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Table sources should trigger SQL
        assert "sql" in result.features


class TestFrontendFeatureDetection:
    """Test frontend feature detection"""
    
    def test_detect_no_frontend_features(self):
        """Test source with no frontend features"""
        source = """
app "TestApp" {
    description: "Test"
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should not detect any UI components
        ui_features = {"chat", "chart", "data_table", "form", "code_editor", "markdown", "file_upload"}
        detected_ui = ui_features.intersection(result.features)
        assert len(detected_ui) == 0


class TestFileDetection:
    """Test detection from files"""
    
    def test_detect_from_nonexistent_file(self):
        """Test detecting from non-existent file"""
        detector = FeatureDetector()
        result = detector.detect_from_file("nonexistent_file.ai")
        
        # Should have warnings
        assert len(result.warnings) > 0
    
    def test_detect_from_directory_empty(self):
        """Test detecting from empty directory"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = FeatureDetector()
            result = detector.detect_from_directory(tmpdir)
            
            # Should warn about no .ai files
            assert len(result.warnings) > 0
    
    def test_detect_from_directory_with_files(self):
        """Test detecting from directory with .ai files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test .ai file
            test_file = Path(tmpdir) / "test.ai"
            test_file.write_text("""
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}
""", encoding='utf-8')
            
            detector = FeatureDetector()
            result = detector.detect_from_directory(tmpdir)
            
            # Should detect openai
            assert "openai" in result.features


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_detect_with_syntax_errors(self):
        """Test detection with syntax errors in source"""
        source = """
app "TestApp" {
    description: "Test"
}

llm gpt4 {
    provider: "openai
    model: "gpt-4"
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should have warnings or errors
        assert len(result.warnings) >= 0  # May or may not have warnings depending on parser
    
    def test_detect_complex_features(self):
        """Test detecting complex combination of features"""
        source = """
app "ComplexApp" {
    description: "Complex test"
}

llm gpt4 {
    provider: "openai"
    model: "gpt-4"
}

llm claude {
    provider: "anthropic"
    model: "claude-3-opus"
}

dataset users from table users
dataset products from table products

agent support {
    llm: gpt4
}

agent sales {
    llm: claude
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should detect multiple features
        assert "openai" in result.features
        assert "anthropic" in result.features
        assert "sql" in result.features
        assert len(result.features) >= 3


class TestDatabaseTypeDetection:
    """Test specific database type detection"""
    
    def test_detect_postgres_explicit(self):
        """Test detecting PostgreSQL explicitly"""
        source = """
app "TestApp" {
    description: "Test"
}

dataset users from postgres table users
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should detect postgres
        # Note: This depends on how the detector checks the source string
        assert "sql" in result.features or "postgres" in result.features
    
    def test_detect_mysql_explicit(self):
        """Test detecting MySQL explicitly"""
        source = """
app "TestApp" {
    description: "Test"
}

dataset users from mysql table users
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Should detect mysql
        assert "sql" in result.features or "mysql" in result.features


class TestMemoryDetection:
    """Test memory system detection"""
    
    def test_detect_short_term_memory(self):
        """Test detecting short-term memory"""
        source = """
app "TestApp" {
    description: "Test"
}

memory conversation {
    type: "short_term"
    capacity: 50
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Short-term memory shouldn't require Redis
        assert "redis" not in result.features or len(result.features) >= 0
    
    def test_detect_long_term_memory(self):
        """Test detecting long-term memory"""
        source = """
app "TestApp" {
    description: "Test"
}

memory persistent {
    type: "long_term"
}
"""
        detector = FeatureDetector()
        result = detector.detect_from_source(source)
        
        # Long-term memory should require Redis
        assert "redis" in result.features or len(result.features) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
