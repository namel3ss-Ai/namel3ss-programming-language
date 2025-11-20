"""
Tests for dataset adapter factory and dataset adapters.

Tests CSV, JSON, inline, and SQL adapter creation, schema inference,
error handling, and fact generation.
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from namel3ss.ast.datasets import Dataset, DatasetSchemaField
from namel3ss.codegen.backend.core.runtime.dataset_adapter_factory import (
    create_adapter_registry,
    create_dataset_adapter,
    _infer_schema_from_records,
)
from namel3ss.codegen.backend.core.runtime.logic_adapters import DatasetAdapter
from namel3ss.ast.application import App


class TestSchemaInference:
    """Test schema inference from records."""
    
    def test_infer_schema_basic_types(self):
        """Test inference of basic types from records."""
        records = [
            {"id": 1, "name": "Alice", "active": True, "score": 95.5},
            {"id": 2, "name": "Bob", "active": False, "score": 87.3},
        ]
        
        schema = _infer_schema_from_records(records, None)
        
        assert len(schema) == 4
        field_map = {f.name: f.dtype for f in schema}
        assert field_map["id"] == "int"
        assert field_map["name"] == "string"
        assert field_map["active"] == "bool"
        assert field_map["score"] == "float"
    
    def test_infer_schema_empty_records(self):
        """Test schema inference with empty records."""
        schema = _infer_schema_from_records([], None)
        assert schema == []
    
    def test_infer_schema_with_explicit_schema(self):
        """Test that explicit schema is used when provided."""
        records = [{"id": 1, "name": "Alice"}]
        explicit_schema = [
            DatasetSchemaField(name="id", dtype="string"),  # Override type
            DatasetSchemaField(name="name", dtype="string"),
        ]
        
        schema = _infer_schema_from_records(records, explicit_schema)
        
        assert len(schema) == 2
        assert schema[0].dtype == "string"  # Explicit type used
        assert schema[1].dtype == "string"
    
    def test_infer_schema_missing_values(self):
        """Test schema inference with missing values."""
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2},  # Missing 'name'
        ]
        
        schema = _infer_schema_from_records(records, None)
        
        # Should infer from first record with the field
        assert len(schema) == 2
        field_map = {f.name: f.dtype for f in schema}
        assert field_map["id"] == "int"
        assert field_map["name"] == "string"


class TestCSVAdapter:
    """Test CSV dataset adapter."""
    
    def test_create_csv_adapter_basic(self):
        """Test creating adapter from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "age"])
            writer.writeheader()
            writer.writerow({"id": "1", "name": "Alice", "age": "30"})
            writer.writerow({"id": "2", "name": "Bob", "age": "25"})
            csv_path = f.name
        
        try:
            dataset = Dataset(
                name="users",
                source_type="csv",
                source=csv_path,
            )
            
            adapter = create_dataset_adapter(dataset)
            
            assert adapter is not None
            assert isinstance(adapter, DatasetAdapter)
            assert adapter.dataset_name == "users"
            assert len(adapter.records) == 2
            assert adapter.records[0]["name"] == "Alice"
            assert adapter.records[1]["name"] == "Bob"
        finally:
            Path(csv_path).unlink()
    
    def test_create_csv_adapter_with_schema(self):
        """Test CSV adapter with explicit schema."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "value"])
            writer.writeheader()
            writer.writerow({"id": "1", "value": "100"})
            csv_path = f.name
        
        try:
            dataset = Dataset(
                name="data",
                source_type="csv",
                source=csv_path,
                schema=[
                    DatasetSchemaField(name="id", dtype="int"),
                    DatasetSchemaField(name="value", dtype="float"),
                ],
            )
            
            adapter = create_dataset_adapter(dataset)
            
            assert adapter is not None
            assert len(adapter.schema) == 2
            assert adapter.schema[0].dtype == "int"
            assert adapter.schema[1].dtype == "float"
        finally:
            Path(csv_path).unlink()
    
    def test_create_csv_adapter_missing_file(self):
        """Test CSV adapter with missing file."""
        dataset = Dataset(
            name="missing",
            source_type="csv",
            source="/nonexistent/file.csv",
        )
        
        # Should return an adapter with empty records rather than raising
        adapter = create_dataset_adapter(dataset)
        assert adapter is not None
        assert len(adapter.records) == 0


class TestJSONAdapter:
    """Test JSON dataset adapter."""
    
    def test_create_json_adapter_basic(self):
        """Test creating adapter from JSON file."""
        data = [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name
        
        try:
            dataset = Dataset(
                name="users",
                source_type="json",
                source=json_path,
            )
            
            adapter = create_dataset_adapter(dataset)
            
            assert adapter is not None
            assert isinstance(adapter, DatasetAdapter)
            assert adapter.dataset_name == "users"
            assert len(adapter.records) == 2
            assert adapter.records[0]["name"] == "Alice"
            assert adapter.records[0]["active"] is True
        finally:
            Path(json_path).unlink()
    
    def test_create_json_adapter_missing_file(self):
        """Test JSON adapter with missing file."""
        dataset = Dataset(
            name="missing",
            source_type="json",
            source="/nonexistent/file.json",
        )
        
        adapter = create_dataset_adapter(dataset)
        assert adapter is not None
        assert len(adapter.records) == 0


class TestInlineAdapter:
    """Test inline dataset adapter."""
    
    def test_create_inline_adapter_basic(self):
        """Test creating adapter from inline records."""
        dataset = Dataset(
            name="products",
            source_type="inline",
            source="",
            metadata={
                "records": [
                    {"id": 1, "name": "Widget", "price": 9.99},
                    {"id": 2, "name": "Gadget", "price": 19.99},
                ]
            },
        )
        
        adapter = create_dataset_adapter(dataset)
        
        assert adapter is not None
        assert isinstance(adapter, DatasetAdapter)
        assert adapter.dataset_name == "products"
        assert len(adapter.records) == 2
        assert adapter.records[0]["name"] == "Widget"
        assert adapter.records[1]["price"] == 19.99
    
    def test_create_inline_adapter_no_records(self):
        """Test inline adapter with no records in metadata."""
        dataset = Dataset(
            name="empty",
            source_type="inline",
            source="",
            metadata={},
        )
        
        adapter = create_dataset_adapter(dataset)
        assert adapter is not None
        assert len(adapter.records) == 0
    
    def test_create_inline_adapter_with_schema(self):
        """Test inline adapter with explicit schema."""
        dataset = Dataset(
            name="typed",
            source_type="inline",
            source="",
            schema=[
                DatasetSchemaField(name="id", dtype="int"),
                DatasetSchemaField(name="value", dtype="string"),
            ],
            metadata={
                "records": [
                    {"id": 1, "value": "test"},
                ]
            },
        )
        
        adapter = create_dataset_adapter(dataset)
        
        assert adapter is not None
        assert len(adapter.schema) == 2
        assert adapter.schema[0].name == "id"
        assert adapter.schema[0].dtype == "int"


class TestSQLAdapter:
    """Test SQL dataset adapter (placeholder implementation)."""
    
    def test_create_sql_adapter_returns_empty(self):
        """Test that SQL adapter returns empty dataset (not implemented)."""
        dataset = Dataset(
            name="sql_data",
            source_type="sql",
            source="SELECT * FROM users",
        )
        
        # Should return empty adapter with warning logged
        adapter = create_dataset_adapter(dataset)
        assert adapter is not None
        assert len(adapter.records) == 0


class TestAdapterRegistry:
    """Test adapter registry creation from App."""
    
    def test_create_adapter_registry_empty_app(self):
        """Test registry creation with no datasets."""
        app = App(name="test")
        
        registry = create_adapter_registry(app)
        
        assert registry is not None
        assert len(registry.adapters) == 0
    
    def test_create_adapter_registry_with_inline_datasets(self):
        """Test registry creation with inline datasets."""
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="users",
                    source_type="inline",
                    source="",
                    metadata={"records": [{"id": 1, "name": "Alice"}]},
                ),
                Dataset(
                    name="products",
                    source_type="inline",
                    source="",
                    metadata={"records": [{"id": 1, "title": "Widget"}]},
                ),
            ],
        )
        
        registry = create_adapter_registry(app)
        
        assert registry is not None
        assert len(registry.adapters) == 2
        assert "users" in registry.adapters
        assert "products" in registry.adapters
    
    def test_create_adapter_registry_mixed_sources(self):
        """Test registry with mixed dataset source types."""
        # Create a temporary CSV for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()
            writer.writerow({"id": "1", "name": "Test"})
            csv_path = f.name
        
        try:
            app = App(
                name="test",
                datasets=[
                    Dataset(
                        name="csv_data",
                        source_type="csv",
                        source=csv_path,
                    ),
                    Dataset(
                        name="inline_data",
                        source_type="inline",
                        source="",
                        metadata={"records": [{"id": 2, "name": "Inline"}]},
                    ),
                    Dataset(
                        name="sql_data",
                        source_type="sql",
                        source="SELECT * FROM test",
                    ),
                ],
            )
            
            registry = create_adapter_registry(app)
            
            # All three should be registered, even if SQL is empty
            assert len(registry.adapters) == 3
            assert "csv_data" in registry.adapters
            assert "inline_data" in registry.adapters
            assert "sql_data" in registry.adapters
        finally:
            Path(csv_path).unlink()


class TestDatasetAdapterFacts:
    """Test fact generation from dataset adapters."""
    
    def test_adapter_generates_row_predicates(self):
        """Test that adapter generates row_* predicates."""
        dataset = Dataset(
            name="users",
            source_type="inline",
            source="",
            metadata={
                "records": [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                ]
            },
        )
        
        adapter = create_dataset_adapter(dataset)
        predicates = adapter.get_predicates()
        
        # Should have row_users predicate
        assert "row_users/1" in predicates
    
    def test_adapter_generates_field_predicates(self):
        """Test that adapter generates field_* predicates."""
        dataset = Dataset(
            name="products",
            source_type="inline",
            source="",
            metadata={
                "records": [
                    {"id": 1, "name": "Widget", "price": 9.99},
                ]
            },
        )
        
        adapter = create_dataset_adapter(dataset)
        predicates = adapter.get_predicates()
        
        # Should have field predicates for each field
        assert "field_products/3" in predicates
    
    def test_adapter_get_facts(self):
        """Test getting all facts from adapter."""
        dataset = Dataset(
            name="data",
            source_type="inline",
            source="",
            metadata={
                "records": [
                    {"id": 1, "value": "test"},
                ]
            },
        )
        
        adapter = create_dataset_adapter(dataset)
        facts = list(adapter.get_facts())
        
        # Should have facts for row and fields
        assert len(facts) > 0
        
        # Check that facts are in the expected format
        # row_data(RowID) and field_data(RowID, FieldName, Value)
        row_facts = [f for f in facts if f.head.functor == "row_data"]
        field_facts = [f for f in facts if f.head.functor == "field_data"]
        
        assert len(row_facts) > 0
        assert len(field_facts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
