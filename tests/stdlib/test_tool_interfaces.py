"""Tests for tool interface standard library components."""

import pytest
from namel3ss.stdlib.tools import (
    ToolCategory,
    ToolInterface,
    HTTPToolSpec,
    DatabaseToolSpec, 
    VectorSearchToolSpec,
    STANDARD_TOOL_SPECS,
    ToolValidationError,
    get_tool_spec,
    list_tool_categories,
    validate_tool_config,
    validate_tool_config_strict,
    suggest_tool_config,
    get_tool_template,
)


class TestToolCategories:
    """Test tool category definitions."""
    
    def test_category_enum_values(self):
        """Test that category enum has expected values."""
        expected_categories = {"http", "database", "vector_search", "python", "custom"}
        actual_categories = {cat.value for cat in ToolCategory}
        assert expected_categories.issubset(actual_categories)
    
    def test_standard_specs_coverage(self):
        """Test that core categories have specifications."""
        core_categories = [ToolCategory.HTTP, ToolCategory.DATABASE, ToolCategory.VECTOR_SEARCH]
        
        for category in core_categories:
            assert category in STANDARD_TOOL_SPECS
            spec = STANDARD_TOOL_SPECS[category]
            assert isinstance(spec, ToolInterface)
            assert spec.category == category
    
    def test_list_categories(self):
        """Test listing all category names."""
        categories = list_tool_categories()
        assert "http" in categories
        assert "database" in categories
        assert "vector_search" in categories


class TestHTTPToolSpec:
    """Test HTTP tool specification and validation."""
    
    def test_http_spec_fields(self):
        """Test HTTP tool required and optional fields."""
        spec = get_tool_spec(ToolCategory.HTTP)
        assert isinstance(spec, HTTPToolSpec)
        
        # Check required fields
        assert "method" in spec.required_fields
        assert "url" in spec.required_fields
        assert "description" in spec.required_fields
        
        # Check optional fields
        assert "headers" in spec.optional_fields
        assert "timeout" in spec.optional_fields
    
    def test_http_valid_config(self):
        """Test validation of valid HTTP tool config."""
        config = {
            "method": "POST",
            "url": "https://api.example.com/endpoint",
            "description": "Example API call",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30
        }
        
        errors = validate_tool_config("http", config)
        assert not errors
    
    def test_http_missing_required(self):
        """Test validation fails for missing required fields."""
        config = {"description": "Incomplete config"}
        
        errors = validate_tool_config("http", config)
        assert "method" in errors
        assert "url" in errors
    
    def test_http_invalid_method(self):
        """Test validation of invalid HTTP methods."""
        config = {
            "method": "INVALID",
            "url": "https://api.example.com",
            "description": "Test"
        }
        
        errors = validate_tool_config("http", config)
        assert "method" in errors
    
    def test_http_invalid_url(self):
        """Test validation of invalid URLs."""
        config = {
            "method": "GET",
            "url": "not-a-url",
            "description": "Test"
        }
        
        errors = validate_tool_config("http", config)
        assert "url" in errors
    
    def test_http_invalid_timeout(self):
        """Test validation of invalid timeout values."""
        config = {
            "method": "GET",
            "url": "https://api.example.com",
            "description": "Test",
            "timeout": -1
        }
        
        errors = validate_tool_config("http", config)
        assert "timeout" in errors
    
    def test_http_invalid_headers(self):
        """Test validation of invalid headers."""
        config = {
            "method": "GET",
            "url": "https://api.example.com",
            "description": "Test",
            "headers": "invalid"  # Should be dict
        }
        
        errors = validate_tool_config("http", config)
        assert "headers" in errors
        
        config["headers"] = {123: "value"}  # Non-string key
        errors = validate_tool_config("http", config)
        assert "headers" in errors


class TestDatabaseToolSpec:
    """Test database tool specification and validation."""
    
    def test_database_spec_fields(self):
        """Test database tool required and optional fields."""
        spec = get_tool_spec(ToolCategory.DATABASE)
        assert isinstance(spec, DatabaseToolSpec)
        
        # Check required fields
        assert "connection" in spec.required_fields
        assert "query_type" in spec.required_fields
        assert "description" in spec.required_fields
        
        # Check optional fields
        assert "query" in spec.optional_fields
        assert "parameters" in spec.optional_fields
        assert "result_limit" in spec.optional_fields
    
    def test_database_valid_config(self):
        """Test validation of valid database tool config."""
        config = {
            "connection": "main_db",
            "query_type": "select",
            "description": "Get user data",
            "query": "SELECT * FROM users WHERE id = ?",
            "parameters": ["user_id"],
            "result_limit": 100,
            "timeout": 30
        }
        
        errors = validate_tool_config("database", config)
        assert not errors
    
    def test_database_invalid_query_type(self):
        """Test validation of invalid query types."""
        config = {
            "connection": "db",
            "query_type": "invalid",
            "description": "Test"
        }
        
        errors = validate_tool_config("database", config)
        assert "query_type" in errors
    
    def test_database_invalid_result_limit(self):
        """Test validation of invalid result limits."""
        config = {
            "connection": "db",
            "query_type": "select",
            "description": "Test",
            "result_limit": -1
        }
        
        errors = validate_tool_config("database", config)
        assert "result_limit" in errors


class TestVectorSearchToolSpec:
    """Test vector search tool specification and validation."""
    
    def test_vector_spec_fields(self):
        """Test vector search tool required and optional fields."""
        spec = get_tool_spec(ToolCategory.VECTOR_SEARCH)
        assert isinstance(spec, VectorSearchToolSpec)
        
        # Check required fields
        assert "index_name" in spec.required_fields
        assert "description" in spec.required_fields
        
        # Check optional fields
        assert "query_vector" in spec.optional_fields
        assert "query_text" in spec.optional_fields
        assert "top_k" in spec.optional_fields
    
    def test_vector_valid_config_with_text(self):
        """Test validation with text query."""
        config = {
            "index_name": "documents",
            "description": "Search documents",
            "query_text": "machine learning",
            "top_k": 5
        }
        
        errors = validate_tool_config("vector_search", config)
        assert not errors
    
    def test_vector_valid_config_with_vector(self):
        """Test validation with vector query."""
        config = {
            "index_name": "embeddings",
            "description": "Vector similarity search",
            "query_vector": [0.1, 0.2, 0.3, 0.4],
            "top_k": 10
        }
        
        errors = validate_tool_config("vector_search", config)
        assert not errors
    
    def test_vector_missing_query(self):
        """Test validation fails without query input."""
        config = {
            "index_name": "documents", 
            "description": "Search"
            # Missing both query_text and query_vector
        }
        
        errors = validate_tool_config("vector_search", config)
        assert "query_input" in errors
    
    def test_vector_invalid_top_k(self):
        """Test validation of invalid top_k values."""
        config = {
            "index_name": "documents",
            "description": "Search",
            "query_text": "test",
            "top_k": 0  # Invalid
        }
        
        errors = validate_tool_config("vector_search", config)
        assert "top_k" in errors
        
        config["top_k"] = 2000  # Too large
        errors = validate_tool_config("vector_search", config)
        assert "top_k" in errors
    
    def test_vector_invalid_threshold(self):
        """Test validation of invalid similarity thresholds."""
        config = {
            "index_name": "documents",
            "description": "Search",
            "query_text": "test",
            "similarity_threshold": 1.5  # > 1.0
        }
        
        errors = validate_tool_config("vector_search", config)
        assert "similarity_threshold" in errors
    
    def test_vector_invalid_query_vector(self):
        """Test validation of invalid query vectors."""
        config = {
            "index_name": "documents",
            "description": "Search", 
            "query_vector": "not a list"
        }
        
        errors = validate_tool_config("vector_search", config)
        assert "query_vector" in errors
        
        config["query_vector"] = ["string", "values"]  # Non-numeric
        errors = validate_tool_config("vector_search", config)
        assert "query_vector" in errors


class TestToolValidation:
    """Test general tool validation functionality."""
    
    def test_get_tool_spec_by_enum(self):
        """Test getting spec by enum value."""
        spec = get_tool_spec(ToolCategory.HTTP)
        assert spec.category == ToolCategory.HTTP
    
    def test_get_tool_spec_by_string(self):
        """Test getting spec by string value."""
        spec = get_tool_spec("http")
        assert spec.category == ToolCategory.HTTP
    
    def test_get_tool_spec_invalid(self):
        """Test error on invalid category."""
        with pytest.raises(ValueError, match="Unknown tool category"):
            get_tool_spec("invalid_category")
    
    def test_get_tool_spec_no_standard(self):
        """Test error for category without standard spec."""
        with pytest.raises(ValueError, match="No standard specification"):
            get_tool_spec(ToolCategory.PYTHON)
    
    def test_strict_validation(self):
        """Test strict validation mode."""
        valid_config = {
            "method": "GET",
            "url": "https://api.example.com",
            "description": "Test API"
        }
        
        # Should pass with valid config
        validate_tool_config_strict("http", valid_config)
        
        # Should raise exception with invalid config
        invalid_config = {"description": "Incomplete"}
        with pytest.raises(ToolValidationError):
            validate_tool_config_strict("http", invalid_config)
    
    def test_invalid_category_validation(self):
        """Test validation with invalid category name."""
        with pytest.raises(ToolValidationError, match="Unknown tool category"):
            validate_tool_config_strict("invalid", {})


class TestToolSuggestions:
    """Test tool configuration suggestion functionality."""
    
    def test_suggest_http_config(self):
        """Test HTTP tool configuration suggestions."""
        config = suggest_tool_config("http")
        
        assert "method" in config
        assert "timeout" in config
        assert "headers" in config
        
        # Should be valid when required fields added
        config.update({
            "url": "https://api.example.com",
            "description": "Test API"
        })
        errors = validate_tool_config("http", config)
        assert not errors
    
    def test_suggest_database_config(self):
        """Test database tool configuration suggestions."""
        config = suggest_tool_config("database")
        
        assert "query_type" in config
        assert "timeout" in config
        assert "result_limit" in config
        
        # Should be valid when required fields added
        config.update({
            "connection": "main_db",
            "description": "Query database"
        })
        errors = validate_tool_config("database", config)
        assert not errors
    
    def test_suggest_vector_config(self):
        """Test vector search tool configuration suggestions."""
        config = suggest_tool_config("vector_search")
        
        assert "top_k" in config
        assert "similarity_threshold" in config
        
        # Should be valid when required fields added
        config.update({
            "index_name": "documents",
            "description": "Search documents",
            "query_text": "test query"
        })
        errors = validate_tool_config("vector_search", config)
        assert not errors
    
    def test_suggest_with_overrides(self):
        """Test suggestions with custom overrides."""
        config = suggest_tool_config("http", timeout=60, custom_field="value")
        
        assert config["timeout"] == 60
        assert config["custom_field"] == "value"
    
    def test_get_tool_template(self):
        """Test tool template generation."""
        template = get_tool_template("http")
        
        assert "tool" in template
        assert "method" in template
        assert "url" in template
        assert "description" in template


class TestToolIntegration:
    """Test integration scenarios with tool configurations."""
    
    def test_realistic_tool_configs(self):
        """Test realistic tool configurations."""
        configs = [
            # REST API integration
            ("http", {
                "method": "POST",
                "url": "https://api.stripe.com/v1/charges",
                "description": "Create payment charge",
                "headers": {
                    "Authorization": "Bearer sk_...",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                "timeout": 10
            }),
            
            # Database query
            ("database", {
                "connection": "analytics_db",
                "query_type": "select",
                "description": "Get user analytics",
                "query": "SELECT * FROM user_events WHERE user_id = ? AND date >= ?",
                "parameters": ["user_id", "start_date"],
                "result_limit": 1000,
                "timeout": 30
            }),
            
            # Vector search
            ("vector_search", {
                "index_name": "product_embeddings",
                "description": "Find similar products",
                "query_text": "comfortable running shoes",
                "top_k": 20,
                "similarity_threshold": 0.7,
                "filters": {"category": "footwear", "in_stock": True}
            })
        ]
        
        for category, config in configs:
            errors = validate_tool_config(category, config)
            assert not errors, f"Realistic config for {category} failed: {errors}"
    
    def test_tool_evolution(self):
        """Test tool configuration evolution scenarios."""
        # Start with basic HTTP tool
        basic_config = suggest_tool_config("http")
        basic_config.update({
            "url": "https://api.example.com/v1",
            "description": "Basic API call"
        })
        
        # Evolve to production config
        prod_config = basic_config.copy()
        prod_config.update({
            "headers": {"Authorization": "Bearer token"},
            "timeout": 5,
            "retry_config": {"max_retries": 3}
        })
        
        # Both should be valid
        assert not validate_tool_config("http", basic_config)
        assert not validate_tool_config("http", prod_config)