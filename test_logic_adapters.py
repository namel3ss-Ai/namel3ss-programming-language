#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test logic adapters for datasets and models."""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from namel3ss.codegen.backend.core.runtime.logic_adapters import (
    AdapterRegistry,
    DatasetAdapter,
    ModelAdapter,
)
from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngine
from namel3ss.ast.logic import LogicAtom, LogicStruct, LogicVar


def test_dataset_adapter():
    """Test dataset adapter generates correct facts."""
    print("Testing dataset adapter...")
    
    # Create a simple dataset
    schema = {"id": "int", "name": "str", "age": "int", "active": "bool"}
    records = [
        {"id": 1, "name": "Alice", "age": 30, "active": True},
        {"id": 2, "name": "Bob", "age": 25, "active": False},
        {"id": 3, "name": "Charlie", "age": 35, "active": True},
    ]
    
    adapter = DatasetAdapter("users", schema, records)
    
    # Check predicates
    predicates = adapter.get_predicates()
    assert "row_users" in predicates
    assert "field_users" in predicates
    print(f"  ✓ Predicates: {predicates}")
    
    # Generate row facts
    row_facts = list(adapter._generate_row_facts())
    assert len(row_facts) == 3
    print(f"  ✓ Generated {len(row_facts)} row facts")
    
    # Check first row fact structure
    first_fact = row_facts[0]
    assert first_fact.head.functor == "row_users"
    assert len(first_fact.head.args) == 5  # row_id + 4 fields
    print(f"  ✓ Row fact: {first_fact}")
    
    # Generate field facts
    field_facts = list(adapter._generate_field_facts())
    assert len(field_facts) == 12  # 3 rows * 4 fields
    print(f"  ✓ Generated {len(field_facts)} field facts")
    
    print("Dataset adapter tests passed!\n")


def test_model_adapter():
    """Test model adapter generates correct facts."""
    print("Testing model adapter...")
    
    # Create model metadata
    models = [
        {
            "name": "classifier_v1",
            "version": "1.0",
            "type": "classification",
            "metrics": {"accuracy": 0.95, "f1": 0.93},
            "params": {"learning_rate": 0.001, "batch_size": 32},
        },
        {
            "name": "regressor_v1",
            "version": "1.0",
            "type": "regression",
            "metrics": {"mse": 0.15, "r2": 0.88},
            "params": {"learning_rate": 0.01, "hidden_units": 64},
        },
    ]
    
    adapter = ModelAdapter(models)
    
    # Check predicates
    predicates = adapter.get_predicates()
    assert "model" in predicates
    assert "model_metric" in predicates
    assert "model_param" in predicates
    print(f"  ✓ Predicates: {predicates}")
    
    # Generate model facts
    model_facts = list(adapter._generate_model_facts())
    assert len(model_facts) == 2
    print(f"  ✓ Generated {len(model_facts)} model facts")
    print(f"    {model_facts[0]}")
    
    # Generate metric facts
    metric_facts = list(adapter._generate_metric_facts())
    assert len(metric_facts) == 4  # 2 models * 2 metrics each
    print(f"  ✓ Generated {len(metric_facts)} metric facts")
    
    # Generate param facts
    param_facts = list(adapter._generate_param_facts())
    assert len(param_facts) == 4  # 2 models * 2 params each
    print(f"  ✓ Generated {len(param_facts)} param facts")
    
    print("Model adapter tests passed!\n")


def test_query_with_dataset():
    """Test querying dataset facts with logic engine."""
    print("Testing queries with dataset adapter...")
    
    # Create dataset
    schema = {"name": "str", "age": "int", "active": "bool"}
    records = [
        {"name": "Alice", "age": 30, "active": True},
        {"name": "Bob", "age": 25, "active": False},
        {"name": "Charlie", "age": 35, "active": True},
    ]
    
    adapter = DatasetAdapter("users", schema, records)
    
    # Get all facts
    all_facts = list(adapter.get_facts())
    
    # Query: field_users(RowId, name, Name)? - Find all names
    goal = LogicStruct(
        functor="field_users",
        args=[
            LogicVar(name="RowId"),
            LogicAtom(value="name"),
            LogicVar(name="Name"),
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([goal], all_facts, []))
    
    assert len(solutions) == 3
    names = [sol.apply(LogicVar(name="Name")).value for sol in solutions]
    assert "Alice" in names
    assert "Bob" in names
    assert "Charlie" in names
    
    print(f"  ✓ Found {len(solutions)} users:")
    for sol in solutions:
        name = sol.apply(LogicVar(name="Name"))
        print(f"    - {name}")
    
    print("Dataset query tests passed!\n")


def test_query_with_models():
    """Test querying model facts with logic engine."""
    print("Testing queries with model adapter...")
    
    # Create models
    models = [
        {
            "name": "classifier_v1",
            "version": "1.0",
            "type": "classification",
            "metrics": {"accuracy": 0.95, "f1": 0.93},
        },
        {
            "name": "classifier_v2",
            "version": "2.0",
            "type": "classification",
            "metrics": {"accuracy": 0.97, "f1": 0.96},
        },
        {
            "name": "regressor_v1",
            "version": "1.0",
            "type": "regression",
            "metrics": {"mse": 0.15, "r2": 0.88},
        },
    ]
    
    adapter = ModelAdapter(models)
    all_facts = list(adapter.get_facts())
    
    # Query: model(Name, _, classification)? - Find all classification models
    goal = LogicStruct(
        functor="model",
        args=[
            LogicVar(name="Name"),
            LogicVar(name="_Version"),  # Don't care about version
            LogicAtom(value="classification"),
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([goal], all_facts, []))
    
    assert len(solutions) == 2
    model_names = [sol.apply(LogicVar(name="Name")).value for sol in solutions]
    assert "classifier_v1" in model_names
    assert "classifier_v2" in model_names
    
    print(f"  ✓ Found {len(solutions)} classification models:")
    for sol in solutions:
        name = sol.apply(LogicVar(name="Name"))
        print(f"    - {name}")
    
    print("Model query tests passed!\n")


def test_adapter_registry():
    """Test adapter registry."""
    print("Testing adapter registry...")
    
    registry = AdapterRegistry()
    
    # Register dataset adapter
    dataset_adapter = DatasetAdapter(
        "users",
        {"name": "str", "age": "int"},
        [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    )
    registry.register("users", dataset_adapter)
    
    # Register model adapter
    model_adapter = ModelAdapter([
        {"name": "model1", "version": "1.0", "type": "classification", "metrics": {}}
    ])
    registry.register("models", model_adapter)
    
    # Check retrieval
    assert registry.get_adapter("users") is dataset_adapter
    assert registry.get_adapter("models") is model_adapter
    assert registry.get_adapter("nonexistent") is None
    print("  ✓ Registry stores and retrieves adapters correctly")
    
    # Get all facts
    all_facts = registry.get_all_facts()
    assert len(all_facts) > 0
    print(f"  ✓ Registry provides {len(all_facts)} total facts from all adapters")
    
    print("Adapter registry tests passed!\n")


if __name__ == '__main__':
    test_dataset_adapter()
    test_model_adapter()
    test_query_with_dataset()
    test_query_with_models()
    test_adapter_registry()
    print("=" * 50)
    print("All adapter tests passed!")
