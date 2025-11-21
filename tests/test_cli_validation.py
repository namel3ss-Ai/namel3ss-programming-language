"""
Test suite for CLI validation functions.

Tests all validation functions with edge cases and the must_exist parameter.
"""

import pytest
from pathlib import Path
import tempfile
import os

from namel3ss.cli.validation import (
    validate_path,
    validate_string,
    validate_bool,
    validate_int,
    validate_target_type,
)
from namel3ss.cli.errors import CLIValidationError


class TestValidatePath:
    """Test validate_path() function."""
    
    def test_valid_string_path(self):
        """Test converting string to Path."""
        result = validate_path("/tmp/test.txt")
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test.txt"
    
    def test_valid_path_object(self):
        """Test passing Path object directly."""
        input_path = Path("/tmp/test.txt")
        result = validate_path(input_path)
        assert result == input_path
    
    def test_none_without_allow_none(self):
        """Test that None raises error by default."""
        with pytest.raises(CLIValidationError) as exc_info:
            validate_path(None)
        assert "cannot be None" in str(exc_info.value)
    
    def test_none_with_allow_none(self):
        """Test that None is accepted when allow_none=True."""
        result = validate_path(None, allow_none=True)
        assert result is None
    
    def test_must_exist_with_existing_file(self, tmp_path):
        """Test must_exist=True with a file that exists."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("content")
        
        result = validate_path(str(test_file), must_exist=True)
        assert result == test_file
        assert result.exists()
    
    def test_must_exist_with_nonexistent_file(self):
        """Test must_exist=True with a file that doesn't exist."""
        nonexistent = Path("/tmp/definitely_does_not_exist_12345.txt")
        
        with pytest.raises(CLIValidationError) as exc_info:
            validate_path(nonexistent, must_exist=True)
        
        assert "does not exist" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)
    
    def test_must_exist_with_directory(self, tmp_path):
        """Test must_exist=True with an existing directory."""
        result = validate_path(str(tmp_path), must_exist=True)
        assert result == tmp_path
        assert result.is_dir()
    
    def test_relative_path(self):
        """Test relative paths are preserved."""
        result = validate_path("./relative/path.txt")
        assert str(result) == "relative/path.txt"
    
    def test_pathlike_object(self, tmp_path):
        """Test os.PathLike objects."""
        # Path is os.PathLike
        result = validate_path(tmp_path)
        assert result == tmp_path


class TestValidateString:
    """Test validate_string() function."""
    
    def test_valid_string(self):
        """Test normal string input."""
        result = validate_string("test_string")
        assert result == "test_string"
    
    def test_none_without_allow_none(self):
        """Test None raises error by default."""
        with pytest.raises(CLIValidationError):
            validate_string(None)
    
    def test_none_with_allow_none(self):
        """Test None is accepted when allow_none=True."""
        result = validate_string(None, allow_none=True)
        assert result is None
    
    def test_empty_string(self):
        """Test empty string is accepted."""
        result = validate_string("")
        assert result == ""
    
    def test_numeric_string(self):
        """Test numeric strings are preserved as strings."""
        result = validate_string("12345")
        assert result == "12345"
        assert isinstance(result, str)


class TestValidateBool:
    """Test validate_bool() function."""
    
    def test_true_value(self):
        """Test various true values."""
        assert validate_bool(True) is True
        assert validate_bool("true") is True
        assert validate_bool("True") is True
        assert validate_bool("yes") is True
        assert validate_bool("1") is True
        assert validate_bool(1) is True
    
    def test_false_value(self):
        """Test various false values."""
        assert validate_bool(False) is False
        assert validate_bool("false") is False
        assert validate_bool("False") is False
        assert validate_bool("no") is False
        assert validate_bool("0") is False
        assert validate_bool(0) is False
    
    def test_none_without_allow_none(self):
        """Test None raises error by default."""
        with pytest.raises(CLIValidationError):
            validate_bool(None)
    
    def test_none_with_allow_none(self):
        """Test None is accepted when allow_none=True."""
        result = validate_bool(None, allow_none=True)
        assert result is None
    
    def test_invalid_value(self):
        """Test invalid boolean value raises error."""
        with pytest.raises(CLIValidationError):
            validate_bool("invalid")


class TestValidateInt:
    """Test validate_int() function."""
    
    def test_valid_integer(self):
        """Test normal integer input."""
        result = validate_int(42)
        assert result == 42
        assert isinstance(result, int)
    
    def test_string_integer(self):
        """Test string that can be parsed as integer."""
        result = validate_int("42")
        assert result == 42
        assert isinstance(result, int)
    
    def test_none_without_allow_none(self):
        """Test None raises error by default."""
        with pytest.raises(CLIValidationError):
            validate_int(None)
    
    def test_none_with_allow_none(self):
        """Test None is accepted when allow_none=True."""
        result = validate_int(None, allow_none=True)
        assert result is None
    
    def test_invalid_string(self):
        """Test non-numeric string raises error."""
        with pytest.raises(CLIValidationError):
            validate_int("not_a_number")
    
    def test_float_truncation(self):
        """Test float is truncated to int."""
        result = validate_int(42.7)
        assert result == 42
        assert isinstance(result, int)
    
    def test_negative_integer(self):
        """Test negative integers."""
        result = validate_int(-42)
        assert result == -42


class TestValidateTargetType:
    """Test validate_target_type() function."""
    
    def test_valid_static(self):
        """Test 'static' target type."""
        result = validate_target_type("static")
        assert result == "static"
    
    def test_valid_react_vite(self):
        """Test 'react-vite' target type."""
        result = validate_target_type("react-vite")
        assert result == "react-vite"
    
    def test_case_insensitive(self):
        """Test target type validation is case-insensitive."""
        assert validate_target_type("Static") == "static"
        assert validate_target_type("REACT-VITE") == "react-vite"
    
    def test_invalid_target(self):
        """Test invalid target type raises error."""
        with pytest.raises(CLIValidationError) as exc_info:
            validate_target_type("invalid_target")
        
        assert "invalid_target" in str(exc_info.value)
        assert "static" in str(exc_info.value)
        assert "react-vite" in str(exc_info.value)
    
    def test_none_value(self):
        """Test None defaults to 'static'."""
        result = validate_target_type(None)
        assert result == "static"


class TestValidationRegression:
    """Regression tests for validation bugs."""
    
    def test_validate_path_must_exist_regression(self, tmp_path):
        """
        Regression test for TypeError: validate_path() got an unexpected keyword argument 'must_exist'.
        
        This was a critical bug in namel3ss build command that prevented builds from working.
        """
        # Create a test file
        test_file = tmp_path / "test.n3"
        test_file.write_text("app 'test' {}")
        
        # This should not raise TypeError
        result = validate_path(str(test_file), must_exist=True)
        assert result == test_file
        
        # Test with nonexistent file
        nonexistent = tmp_path / "nonexistent.n3"
        with pytest.raises(CLIValidationError) as exc_info:
            validate_path(str(nonexistent), must_exist=True)
        
        # Should be CLIValidationError, not TypeError
        assert "does not exist" in str(exc_info.value)
    
    def test_validate_path_allow_none_and_must_exist(self):
        """Test that allow_none and must_exist can be used together."""
        # None should be allowed even with must_exist=True
        result = validate_path(None, allow_none=True, must_exist=True)
        assert result is None
    
    def test_deterministic_path_resolution(self, tmp_path):
        """Test that path resolution is deterministic."""
        # Create multiple files
        files = []
        for i in range(5):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)
        
        # Validate all paths
        results = [validate_path(str(f), must_exist=True) for f in files]
        
        # Results should be in the same order as input
        assert results == files
        
        # Running again should produce same order
        results2 = [validate_path(str(f), must_exist=True) for f in files]
        assert results2 == files
        assert results2 == results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
