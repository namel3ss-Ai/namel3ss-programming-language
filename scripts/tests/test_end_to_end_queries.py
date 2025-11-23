#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end test: parse .ai file and execute queries."""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from namel3ss.lang.grammar import parse_module
from namel3ss.ast import App, KnowledgeModule, LogicQuery
from namel3ss.codegen.backend.core.runtime.logic_adapters import (
    AdapterRegistry,
    DatasetAdapter,
)
from namel3ss.codegen.backend.core.runtime.query_compiler import (
    QueryCompiler,
    QueryContext,
)


def test_end_to_end():
    """Test complete flow: parse -> compile -> execute."""
    print("=" * 60)
    print("END-TO-END TEST: Parse N3 file and execute queries")
    print("=" * 60)
    
    # Parse the test file
    print("\n1. Parsing test_knowledge_parsing.ai...")
    with open('test_knowledge_parsing.ai', 'r') as f:
        code = f.read()
    
    mod = parse_module(code, path="test_knowledge_parsing.ai")
    
    # Find the app
    app = None
    for item in mod.body:
        if isinstance(item, App):
            app = item
            break
    
    assert app is not None, "App not found"
    print(f"   ✓ Parsed app: {app.name}")
    print(f"   ✓ Found {len(app.knowledge_modules)} knowledge modules")
    print(f"   ✓ Found {len(app.queries)} queries")
    
    # Create knowledge module map
    print("\n2. Setting up query context...")
    knowledge_map = {km.name: km for km in app.knowledge_modules}
    
    # Create adapter registry (could add dataset/model adapters here)
    adapter_registry = AdapterRegistry()
    
    # Optionally add a dataset adapter
    user_data = [
        {"name": "Alice", "role": "parent"},
        {"name": "Bob", "role": "child"},
        {"name": "Charlie", "role": "grandchild"},
    ]
    dataset_adapter = DatasetAdapter(
        "users",
        {"name": "str", "role": "str"},
        user_data
    )
    adapter_registry.register("users", dataset_adapter)
    print(f"   ✓ Registered {len(adapter_registry.adapters)} adapters")
    
    # Create query context
    context = QueryContext(
        knowledge_modules=knowledge_map,
        adapter_registry=adapter_registry,
    )
    print(f"   ✓ Context created with {len(knowledge_map)} knowledge modules")
    
    # Create compiler
    compiler = QueryCompiler(context)
    
    # Execute each query
    print("\n3. Executing queries...")
    for query in app.queries:
        print(f"\n   Query: {query.name}")
        print(f"   - Knowledge sources: {query.knowledge_sources}")
        print(f"   - Goals: {[str(g) for g in query.goals]}")
        if query.variables:
            print(f"   - Variables: {query.variables}")
        
        # Compile query
        compiled = compiler.compile_query(query)
        print(f"   - Collected {len(compiled.facts)} facts, {len(compiled.rules)} rules")
        
        # Execute query
        results = compiled.execute_all()
        print(f"   ✓ Found {len(results)} solutions:")
        
        for i, result in enumerate(results, 1):
            print(f"     {i}. {result}")
    
    # Test a specific query expectation
    print("\n4. Validating query results...")
    if app.queries:
        query = app.queries[0]  # find_ancestors query
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        # Should find bob and charlie as ancestors of alice
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        who_values = [r.get("Who") for r in results]
        assert "bob" in who_values, f"Expected 'bob' in results: {who_values}"
        assert "charlie" in who_values, f"Expected 'charlie' in results: {who_values}"
        
        print("   ✓ Query results validated successfully")
    
    print("\n" + "=" * 60)
    print("END-TO-END TEST PASSED!")
    print("=" * 60)


def test_query_with_limit():
    """Test query limit functionality."""
    print("\n" + "=" * 60)
    print("TEST: Query with limit")
    print("=" * 60)
    
    # Create test N3 code with limit
    code = """
app "test".

knowledge numbers:
    fact number(1).
    fact number(2).
    fact number(3).
    fact number(4).
    fact number(5).

query first_two:
    from knowledge numbers:
    goal number(N)
    limit: 2
    variables: N
"""
    
    print("\nParsing test code...")
    mod = parse_module(code, path="test_limit.ai")
    
    # Find app
    app = None
    for item in mod.body:
        if isinstance(item, App):
            app = item
            break
    
    print(f"✓ Found query: {app.queries[0].name} (limit={app.queries[0].limit})")
    
    # Create context and compiler
    knowledge_map = {km.name: km for km in app.knowledge_modules}
    context = QueryContext(knowledge_modules=knowledge_map)
    compiler = QueryCompiler(context)
    
    # Execute query
    query = app.queries[0]
    compiled = compiler.compile_query(query)
    results = compiled.execute_all()
    
    print(f"✓ Results (should be limited to 2): {results}")
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    print("\n" + "=" * 60)
    print("LIMIT TEST PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    test_end_to_end()
    test_query_with_limit()
    print("\n🎉 All end-to-end tests passed!")
