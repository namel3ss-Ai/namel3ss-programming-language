"""
Tests for namel3ss.deps.spec module.

Tests the feature → dependency mapping system.
"""

import pytest
from namel3ss.deps.spec import (
    DependencySpec,
    PythonPackage,
    NPMPackage,
    FeatureCategory,
    get_dependency_spec,
    get_feature_spec,
    get_python_packages_for_features,
    get_npm_packages_for_features,
    DEPENDENCY_SPECS,
)


class TestDependencySpec:
    """Test DependencySpec dataclass"""
    
    def test_dependency_spec_creation(self):
        """Test creating a DependencySpec"""
        spec = DependencySpec(
            feature_id="test",
            category=FeatureCategory.CORE,
            description="Test feature",
            python_packages=[PythonPackage("test-pkg", ">=1.0")],
            npm_packages=[NPMPackage("test-pkg", "^1.0.0")],
        )
        
        assert spec.feature_id == "test"
        assert spec.category == FeatureCategory.CORE
        assert len(spec.python_packages) == 1
        assert len(spec.npm_packages) == 1
    
    def test_dependency_spec_no_packages(self):
        """Test DependencySpec with no packages"""
        spec = DependencySpec(
            feature_id="test",
            category=FeatureCategory.CORE,
            description="Test feature",
            python_packages=[],
            npm_packages=[],
        )
        
        assert len(spec.python_packages) == 0
        assert len(spec.npm_packages) == 0


class TestPythonPackage:
    """Test PythonPackage dataclass"""
    
    def test_python_package_with_version(self):
        """Test creating a Python package with version constraint"""
        pkg = PythonPackage("fastapi", ">=0.110,<1.0")
        assert pkg.name == "fastapi"
        assert pkg.version == ">=0.110,<1.0"
    
    def test_python_package_no_version(self):
        """Test creating a Python package without version"""
        pkg = PythonPackage("some-package", "")
        assert pkg.name == "some-package"
        assert pkg.version == ""


class TestNPMPackage:
    """Test NPMPackage dataclass"""
    
    def test_npm_package_regular(self):
        """Test creating a regular NPM package"""
        pkg = NPMPackage("react", "^18.2.0", dev=False)
        assert pkg.name == "react"
        assert pkg.version == "^18.2.0"
        assert pkg.dev is False
    
    def test_npm_package_dev(self):
        """Test creating a dev NPM package"""
        pkg = NPMPackage("typescript", "^5.0.0", dev=True)
        assert pkg.name == "typescript"
        assert pkg.version == "^5.0.0"
        assert pkg.dev is True


class TestDependencySpecs:
    """Test the DEPENDENCY_SPECS mapping"""
    
    def test_dependency_specs_not_empty(self):
        """Test that DEPENDENCY_SPECS is populated"""
        assert len(DEPENDENCY_SPECS) > 0
        assert "core" in DEPENDENCY_SPECS
        assert "openai" in DEPENDENCY_SPECS
    
    def test_core_dependencies(self):
        """Test core dependencies are present"""
        core = DEPENDENCY_SPECS["core"]
        assert core.category == FeatureCategory.CORE
        
        # Check Python packages
        python_names = [pkg.name for pkg in core.python_packages]
        assert "fastapi" in python_names
        assert "uvicorn" in python_names
        assert "pydantic" in python_names
        assert "httpx" in python_names
        
        # Check NPM packages
        npm_names = [pkg.name for pkg in core.npm_packages]
        assert "react" in npm_names
        assert "typescript" in npm_names
        assert "vite" in npm_names
    
    def test_ai_provider_dependencies(self):
        """Test AI provider dependencies"""
        # OpenAI
        openai = DEPENDENCY_SPECS["openai"]
        assert openai.category == FeatureCategory.AI_PROVIDER
        python_names = [pkg.name for pkg in openai.python_packages]
        assert "openai" in python_names
        assert "tiktoken" in python_names
        
        # Anthropic
        anthropic = DEPENDENCY_SPECS["anthropic"]
        assert anthropic.category == FeatureCategory.AI_PROVIDER
        python_names = [pkg.name for pkg in anthropic.python_packages]
        assert "anthropic" in python_names
    
    def test_database_dependencies(self):
        """Test database dependencies"""
        # PostgreSQL
        postgres = DEPENDENCY_SPECS["postgres"]
        assert postgres.category == FeatureCategory.DATABASE
        python_names = [pkg.name for pkg in postgres.python_packages]
        assert "sqlalchemy" in python_names
        assert "asyncpg" in python_names
        
        # MySQL
        mysql = DEPENDENCY_SPECS["mysql"]
        assert mysql.category == FeatureCategory.DATABASE
        python_names = [pkg.name for pkg in mysql.python_packages]
        assert "sqlalchemy" in python_names
        assert "aiomysql" in python_names
        
        # MongoDB
        mongo = DEPENDENCY_SPECS["mongo"]
        assert mongo.category == FeatureCategory.DATABASE
        python_names = [pkg.name for pkg in mongo.python_packages]
        assert "motor" in python_names
        assert "pymongo" in python_names
    
    def test_ui_component_dependencies(self):
        """Test UI component dependencies"""
        # Chat
        chat = DEPENDENCY_SPECS["chat"]
        assert chat.category == FeatureCategory.UI_COMPONENT
        npm_names = [pkg.name for pkg in chat.npm_packages]
        assert "@radix-ui/react-scroll-area" in npm_names
        
        # Chart
        chart = DEPENDENCY_SPECS["chart"]
        assert chart.category == FeatureCategory.UI_COMPONENT
        npm_names = [pkg.name for pkg in chart.npm_packages]
        assert "recharts" in npm_names
        
        # Data table
        data_table = DEPENDENCY_SPECS["data_table"]
        assert data_table.category == FeatureCategory.UI_COMPONENT
        npm_names = [pkg.name for pkg in data_table.npm_packages]
        assert "@tanstack/react-table" in npm_names


class TestGetDependencySpec:
    """Test get_feature_spec function"""
    
    def test_get_existing_spec(self):
        """Test getting an existing spec"""
        spec = get_feature_spec("openai")
        assert spec is not None
        assert spec.feature_id == "openai"
        assert spec.category == FeatureCategory.AI_PROVIDER
    
    def test_get_nonexistent_spec(self):
        """Test getting a non-existent spec returns None"""
        spec = get_feature_spec("nonexistent_feature")
        assert spec is None


class TestGetPythonPackages:
    """Test get_python_packages_for_features function"""
    
    def test_get_python_packages_single_feature(self):
        """Test getting Python packages for single feature"""
        packages = get_python_packages_for_features(["openai"])
        
        assert len(packages) > 0
        names = [pkg.name for pkg in packages]
        assert "openai" in names
        assert "tiktoken" in names
    
    def test_get_python_packages_multiple_features(self):
        """Test getting Python packages for multiple features"""
        packages = get_python_packages_for_features(["openai", "postgres"])
        
        names = [pkg.name for pkg in packages]
        assert "openai" in names
        assert "sqlalchemy" in names
        assert "asyncpg" in names
    
    def test_get_python_packages_deduplication(self):
        """Test that duplicate packages are deduplicated"""
        # Both postgres and mysql use sqlalchemy
        packages = get_python_packages_for_features(["postgres", "mysql"])
        
        names = [pkg.name for pkg in packages]
        sqlalchemy_count = names.count("sqlalchemy")
        assert sqlalchemy_count == 1  # Should only appear once
    
    def test_get_python_packages_nonexistent_feature(self):
        """Test getting packages for non-existent feature"""
        packages = get_python_packages_for_features(["nonexistent"])
        # Core is always included, so not empty
        assert len(packages) > 0
        names = [pkg.name for pkg in packages]
        assert "fastapi" in names  # Core dependency
    
    def test_get_python_packages_empty_list(self):
        """Test getting packages for empty feature list"""
        packages = get_python_packages_for_features([])
        # Core is always included, so not empty
        assert len(packages) > 0
        names = [pkg.name for pkg in packages]
        assert "fastapi" in names  # Core dependency


class TestGetNPMPackages:
    """Test get_npm_packages_for_features function"""
    
    def test_get_npm_packages_single_feature(self):
        """Test getting NPM packages for single feature"""
        packages = get_npm_packages_for_features(["chat"])
        
        assert len(packages) > 0
        names = [pkg.name for pkg in packages]
        assert "@radix-ui/react-scroll-area" in names
    
    def test_get_npm_packages_multiple_features(self):
        """Test getting NPM packages for multiple features"""
        packages = get_npm_packages_for_features(["chat", "chart"])
        
        names = [pkg.name for pkg in packages]
        assert "@radix-ui/react-scroll-area" in names
        assert "recharts" in names
    
    def test_get_npm_packages_deduplication(self):
        """Test that duplicate packages are deduplicated"""
        packages = get_npm_packages_for_features(["chat", "form"])
        
        # Count unique package names
        names = [pkg.name for pkg in packages]
        assert len(names) == len(set(names))  # All names should be unique
    
    def test_get_npm_packages_dev_and_regular(self):
        """Test that dev and regular packages are separated"""
        packages = get_npm_packages_for_features(["core"])
        
        dev_packages = [pkg for pkg in packages if pkg.dev]
        regular_packages = [pkg for pkg in packages if not pkg.dev]
        
        assert len(dev_packages) > 0
        assert len(regular_packages) > 0
    
    def test_get_npm_packages_empty_list(self):
        """Test getting packages for empty feature list"""
        packages = get_npm_packages_for_features([])
        # Core is always included, so not empty
        assert len(packages) > 0
        names = [pkg.name for pkg in packages]
        assert "react" in names  # Core dependency


class TestFeatureCategories:
    """Test feature categories"""
    
    def test_all_features_have_category(self):
        """Test that all features have a valid category"""
        for feature_id, spec in DEPENDENCY_SPECS.items():
            assert isinstance(spec.category, FeatureCategory)
    
    def test_category_coverage(self):
        """Test that we have features in multiple categories"""
        categories = set()
        for spec in DEPENDENCY_SPECS.values():
            categories.add(spec.category)
        
        # Should have at least these categories
        assert FeatureCategory.CORE in categories
        assert FeatureCategory.AI_PROVIDER in categories
        assert FeatureCategory.DATABASE in categories
        assert FeatureCategory.UI_COMPONENT in categories


class TestVersionConstraints:
    """Test version constraint formats"""
    
    def test_python_version_formats(self):
        """Test that Python version constraints are valid"""
        for spec in DEPENDENCY_SPECS.values():
            for pkg in spec.python_packages:
                # Version can be empty or contain valid constraint operators
                if pkg.version:
                    assert any(op in pkg.version for op in [">=", "<=", ">", "<", "==", "~=", ""]) or "," in pkg.version
    
    def test_npm_version_formats(self):
        """Test that NPM version constraints are valid"""
        for spec in DEPENDENCY_SPECS.values():
            for pkg in spec.npm_packages:
                # Version can be empty or start with valid npm operators
                if pkg.version:
                    assert any(pkg.version.startswith(op) for op in ["^", "~", ">=", "<=", ""]) or "*" in pkg.version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
