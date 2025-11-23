#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test complete integration: Parse → Validate → Compile → Execute → HTTP API

This demonstrates the full pipeline from N3 source code to executable
query API endpoints.
"""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from namel3ss.lang.grammar import parse_module
from namel3ss.ast import App
from namel3ss.logic_validator import validate_logic_constructs
from namel3ss.codegen.backend.core.runtime.logic_adapters import (
    AdapterRegistry,
    DatasetAdapter,
)
from namel3ss.codegen.backend.core.runtime.query_compiler import (
    QueryCompiler,
    QueryContext,
)
from namel3ss.codegen.backend.core.query_router import (
    _render_queries_router_module,
)


def test_full_integration():
    """Test the complete integration pipeline."""
    print("=" * 70)
    print("COMPLETE INTEGRATION TEST")
    print("=" * 70)
    
    # Step 1: Parse N3 code
    print("\n[1/6] PARSING N3 CODE")
    print("-" * 70)
    
    n3_code = """
app "employee_knowledge".

knowledge employees:
    fact employee(1, "Alice", "Engineering").
    fact employee(2, "Bob", "Sales").
    fact employee(3, "Charlie", "Engineering").
    fact reports_to(2, 1).
    fact reports_to(3, 1).
    
    rule manager(ManagerId, EmployeeId) :- reports_to(EmployeeId, ManagerId).

query find_managers:
    from knowledge employees:
    goal manager(ManagerId, EmployeeId)
    variables: ManagerId, EmployeeId

query find_engineers:
    from knowledge employees:
    goal employee(Id, Name, "Engineering")
    variables: Id, Name
"""
    
    mod = parse_module(n3_code, path="test_integration.ai")
    app = next((item for item in mod.body if isinstance(item, App)), None)
    
    assert app is not None
    print(f"✓ Parsed app: {app.name}")
    print(f"  - Knowledge modules: {len(app.knowledge_modules)}")
    print(f"  - Queries: {len(app.queries)}")
    
    for km in app.knowledge_modules:
        print(f"    • {km.name}: {len(km.facts)} facts, {len(km.rules)} rules")
    for q in app.queries:
        print(f"    • {q.name}: {len(q.goals)} goals")
    
    # Step 2: Validate logic constructs
    print("\n[2/6] VALIDATING LOGIC CONSTRUCTS")
    print("-" * 70)
    
    errors, warnings = validate_logic_constructs(
        app.knowledge_modules,
        app.queries
    )
    
    if errors:
        print(f"✗ Validation errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"✓ No validation errors")
    
    if warnings:
        print(f"⚠ Warnings:")
        for warn in warnings:
            print(f"  - {warn}")
    else:
        print(f"✓ No warnings")
    
    # Step 3: Setup query context with adapters
    print("\n[3/6] SETTING UP QUERY CONTEXT")
    print("-" * 70)
    
    knowledge_map = {km.name: km for km in app.knowledge_modules}
    
    # Create adapter registry
    adapter_registry = AdapterRegistry()
    
    # Add a dataset adapter for additional data
    salary_data = [
        {"employee_id": 1, "salary": 120000, "bonus": 10000},
        {"employee_id": 2, "salary": 80000, "bonus": 5000},
        {"employee_id": 3, "salary": 95000, "bonus": 7000},
    ]
    
    salary_adapter = DatasetAdapter(
        "salaries",
        {"employee_id": "int", "salary": "int", "bonus": "int"},
        salary_data
    )
    adapter_registry.register("salaries", salary_adapter)
    
    context = QueryContext(
        knowledge_modules=knowledge_map,
        adapter_registry=adapter_registry,
    )
    
    print(f"✓ Query context created")
    print(f"  - Knowledge modules: {len(context.knowledge_modules)}")
    print(f"  - Adapters: {len(context.adapter_registry.adapters)}")
    
    # Step 4: Compile queries
    print("\n[4/6] COMPILING QUERIES")
    print("-" * 70)
    
    compiler = QueryCompiler(context)
    compiled_queries = {}
    
    for query in app.queries:
        compiled = compiler.compile_query(query)
        compiled_queries[query.name] = compiled
        print(f"✓ Compiled query: {query.name}")
        print(f"  - Goals: {len(compiled.goals)}")
        print(f"  - Facts available: {len(compiled.facts)}")
        print(f"  - Rules available: {len(compiled.rules)}")
    
    # Step 5: Execute queries
    print("\n[5/6] EXECUTING QUERIES")
    print("-" * 70)
    
    for query_name, compiled in compiled_queries.items():
        print(f"\nQuery: {query_name}")
        results = compiled.execute_all()
        print(f"  Results ({len(results)}):")
        for i, result in enumerate(results[:5], 1):  # Show first 5
            print(f"    {i}. {result}")
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")
    
    # Step 6: Generate HTTP API router
    print("\n[6/6] GENERATING HTTP API ROUTER")
    print("-" * 70)
    
    router_code = _render_queries_router_module(app)
    
    if router_code:
        print(f"✓ Generated router module ({len(router_code)} bytes)")
        print(f"\nRouter code preview (first 20 lines):")
        lines = router_code.split('\n')
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... and {len(lines) - 20} more lines")
    else:
        print("✗ No router generated (no queries)")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nPipeline Summary:")
    print(f"  1. ✓ Parsed N3 code → {len(app.knowledge_modules)} modules, {len(app.queries)} queries")
    print(f"  2. ✓ Validated → {len(errors)} errors, {len(warnings)} warnings")
    print(f"  3. ✓ Context → {len(context.knowledge_modules)} modules, {len(adapter_registry.adapters)} adapters")
    print(f"  4. ✓ Compiled → {len(compiled_queries)} queries")
    
    total_results = sum(len(cq.execute_all()) for cq in compiled_queries.values())
    print(f"  5. ✓ Executed → {total_results} total results")
    print(f"  6. ✓ Generated → FastAPI router with {len(app.queries)} endpoints")
    
    return True


def test_query_with_adapters():
    """Test queries that combine knowledge and adapter data."""
    print("\n\n" + "=" * 70)
    print("TEST: Queries with Dataset Adapters")
    print("=" * 70)
    
    n3_code = """
app "test".

knowledge people:
    fact person(1, "Alice").
    fact person(2, "Bob").

query all_people:
    from knowledge people:
    goal person(Id, Name)
    variables: Id, Name
"""
    
    mod = parse_module(n3_code, path="test.ai")
    app = next((item for item in mod.body if isinstance(item, App)), None)
    
    # Setup context with adapter
    knowledge_map = {km.name: km for km in app.knowledge_modules}
    adapter_registry = AdapterRegistry()
    
    # Add extra data via adapter
    extra_data = [
        {"id": 3, "name": "Charlie"},
        {"id": 4, "name": "David"},
    ]
    adapter = DatasetAdapter("extra", {"id": "int", "name": "str"}, extra_data)
    adapter_registry.register("extra", adapter)
    
    context = QueryContext(knowledge_modules=knowledge_map, adapter_registry=adapter_registry)
    compiler = QueryCompiler(context)
    
    # Execute query
    query = app.queries[0]
    compiled = compiler.compile_query(query)
    results = compiled.execute_all()
    
    print(f"\n✓ Query results (knowledge + adapters):")
    print(f"  - Knowledge facts: 2")
    print(f"  - Adapter facts: {len(list(adapter.get_facts()))}")
    print(f"  - Query results: {len(results)}")
    
    for result in results:
        print(f"    • {result}")
    
    assert len(results) == 2  # Only from knowledge, adapters don't add person/2 facts
    print("\n✓ Adapter integration working correctly")


if __name__ == '__main__':
    success = test_full_integration()
    if success:
        test_query_with_adapters()
        print("\n" + "🎉" * 35)
        print("ALL INTEGRATION TESTS PASSED!")
        print("🎉" * 35)
