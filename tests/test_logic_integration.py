"""
Comprehensive integration tests for logic and knowledge system.
Tests complete .n3 programs with knowledge modules and queries.
"""

import sys
import os
from textwrap import dedent

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namel3ss.lang.grammar import parse_module
from namel3ss.logic_validator import LogicValidator
from namel3ss.codegen.backend.core.runtime.query_compiler import QueryContext, QueryCompiler
from namel3ss.codegen.backend.core.runtime.logic_adapters import DatasetAdapter


def test_family_tree():
    """Test a complete family tree knowledge base with queries."""
    print("\n=== Testing Family Tree ===")
    
    code = dedent("""
    app "FamilyTree":
    
    knowledge family:
        # Parent relationships
        fact parent(alice, bob).
        fact parent(alice, charlie).
        fact parent(bob, dave).
        fact parent(charlie, eve).
        fact parent(eve, frank).
        
        # Ancestor rules
        rule ancestor(X, Y) :- parent(X, Y).
        rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        
        # Sibling rule
        rule sibling(X, Y) :- parent(P, X), parent(P, Y).
    
    query find_ancestors:
        from knowledge family:
        goal ancestor(alice, X)
        variables: X
    
    query find_siblings:
        from knowledge family:
        goal sibling(bob, X)
        variables: X
    
    query find_descendants:
        from knowledge family:
        goal ancestor(X, frank)
        variables: X
    """)
    
    # Parse
    ast = parse_module(code)
    
    print(f"✓ Parsed: {len(ast.knowledge_modules)} knowledge module, {len(ast.queries)} queries")
    
    # Validate
    validator = LogicValidator()
    errors = validator.validate_app(ast)
    assert len(errors) == 0, f"Validation errors: {errors}"
    print(f"✓ Validated: 0 errors")
    
    # Compile queries
    context = QueryContext()
    context.add_knowledge_module(ast.knowledge_modules[0])
    compiler = QueryCompiler(context)
    
    compiled_queries = {}
    for query in ast.queries:
        compiled = compiler.compile(query)
        compiled_queries[query.name] = compiled
    
    print(f"✓ Compiled: {len(compiled_queries)} queries")
    
    # Execute queries
    results = {}
    
    # find_ancestors: Should find bob, charlie, dave, eve, frank (all descendants of alice)
    results["find_ancestors"] = list(compiled_queries["find_ancestors"].execute_all())
    descendants = {r["X"].value for r in results["find_ancestors"]}
    assert descendants == {"bob", "charlie", "dave", "eve", "frank"}
    print(f"✓ find_ancestors: Found {len(results['find_ancestors'])} descendants")
    
    # find_siblings: Should find charlie (bob's sibling)
    results["find_siblings"] = list(compiled_queries["find_siblings"].execute_all())
    siblings = {r["X"].value for r in results["find_siblings"]}
    # Note: bob will unify with itself, so we get bob and charlie
    assert "charlie" in siblings
    print(f"✓ find_siblings: Found {len(results['find_siblings'])} siblings")
    
    # find_descendants: Should find alice, eve, charlie (ancestors of frank)
    results["find_descendants"] = list(compiled_queries["find_descendants"].execute_all())
    ancestors = {r["X"].value for r in results["find_descendants"]}
    assert ancestors == {"alice", "eve", "charlie"}
    print(f"✓ find_descendants: Found {len(results['find_descendants'])} ancestors")
    
    print("✅ Family tree test passed!\n")


def test_employee_database():
    """Test employee database with joins and filters."""
    print("\n=== Testing Employee Database ===")
    
    code = dedent("""
    app "EmployeeDB":
    
    knowledge employees:
        # Employee records
        fact employee(1, "Alice", "Engineering").
        fact employee(2, "Bob", "Sales").
        fact employee(3, "Charlie", "Engineering").
        fact employee(4, "Dave", "HR").
        
        # Salary info
        fact salary(1, 100000).
        fact salary(2, 80000).
        fact salary(3, 95000).
        fact salary(4, 75000).
        
        # Manager relationships
        fact reports_to(2, 1).
        fact reports_to(3, 1).
        fact reports_to(4, 2).
        
        # Rules
        rule engineer(Id, Name) :- employee(Id, Name, "Engineering").
        rule manager(ManagerId, EmployeeId) :- reports_to(EmployeeId, ManagerId).
        rule high_earner(Id, Name, Salary) :- 
            employee(Id, Name, Dept), 
            salary(Id, Salary).
    
    query find_engineers:
        from knowledge employees:
        goal engineer(Id, Name)
        variables: Id, Name
    
    query find_managers:
        from knowledge employees:
        goal manager(ManagerId, EmployeeId)
        variables: ManagerId, EmployeeId
    
    query find_earnings:
        from knowledge employees:
        goal high_earner(Id, Name, Salary)
        variables: Id, Name, Salary
        limit: 10
    """)
    
    # Parse
    ast = parse_module(code)
    
    assert len(ast.knowledge_modules) == 1
    assert len(ast.queries) == 3
    print(f"✓ Parsed: {len(ast.queries)} queries")
    
    # Validate
    validator = LogicValidator()
    errors = validator.validate_app(ast)
    assert len(errors) == 0
    print(f"✓ Validated: 0 errors")
    
    # Compile and execute
    context = QueryContext()
    context.add_knowledge_module(ast.knowledge_modules[0])
    compiler = QueryCompiler(context)
    
    # Find engineers
    query = ast.queries[0]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    assert len(results) == 2
    engineers = {(r["Id"].value, r["Name"].value) for r in results}
    assert engineers == {(1, "Alice"), (3, "Charlie")}
    print(f"✓ find_engineers: Found {len(results)} engineers")
    
    # Find managers
    query = ast.queries[1]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    assert len(results) == 3
    print(f"✓ find_managers: Found {len(results)} manager relationships")
    
    # Find high earners
    query = ast.queries[2]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    assert len(results) == 4  # All 4 employees have salary records
    print(f"✓ find_earnings: Found {len(results)} employee earnings")
    
    print("✅ Employee database test passed!\n")


def test_with_adapters():
    """Test knowledge system with dataset adapters."""
    print("\n=== Testing with Dataset Adapters ===")
    
    code = dedent("""
    app "DatasetQuery":
    
    knowledge facts:
        fact category(1, "A").
        fact category(2, "B").
    
    query find_with_salary:
        from knowledge facts:
        from dataset salaries:
        goal category(Id, Cat)
        variables: Id, Cat
    """)
    
    # Parse
    ast = parse_module(code)
    
    print(f"✓ Parsed: {len(ast.queries)} query")
    
    # Validate
    validator = LogicValidator()
    errors = validator.validate_app(ast)
    assert len(errors) == 0
    print(f"✓ Validated: 0 errors")
    
    # Setup context with adapter
    context = QueryContext()
    context.add_knowledge_module(ast.knowledge_modules[0])
    
    # Add mock dataset adapter
    class MockDataset:
        def get_rows(self):
            return [
                {"id": 1, "name": "Alice", "amount": 1000},
                {"id": 2, "name": "Bob", "amount": 2000},
            ]
    
    adapter = DatasetAdapter("salaries", MockDataset())
    context.add_adapter(adapter)
    
    compiler = QueryCompiler(context)
    
    # Compile and execute
    query = ast.queries[0]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    
    assert len(results) == 2
    print(f"✓ find_with_salary: Found {len(results)} results with adapter")
    print(f"✓ Adapter provided {len(list(adapter.get_facts()))} facts")
    
    print("✅ Adapter integration test passed!\n")


def test_complex_rules():
    """Test complex rule patterns."""
    print("\n=== Testing Complex Rules ===")
    
    code = dedent("""
    app "ComplexRules":
    
    knowledge graph:
        # Graph edges
        fact edge(a, b).
        fact edge(b, c).
        fact edge(c, d).
        fact edge(b, e).
        fact edge(e, f).
        
        # Path rules (transitive closure)
        rule path(X, Y) :- edge(X, Y).
        rule path(X, Z) :- edge(X, Y), path(Y, Z).
        
        # Reachability from specific node
        rule reachable_from_a(X) :- path(a, X).
    
    query find_paths:
        from knowledge graph:
        goal path(X, Y)
        variables: X, Y
        limit: 20
    
    query find_reachable:
        from knowledge graph:
        goal reachable_from_a(X)
        variables: X
    """)
    
    # Parse
    ast = parse_module(code)
    
    print(f"✓ Parsed: {len(ast.knowledge_modules)} knowledge module")
    
    # Validate
    validator = LogicValidator()
    errors = validator.validate_app(ast)
    assert len(errors) == 0
    print(f"✓ Validated: 0 errors")
    
    # Compile
    context = QueryContext()
    context.add_knowledge_module(ast.knowledge_modules[0])
    compiler = QueryCompiler(context)
    
    # Find all paths
    query = ast.queries[0]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    print(f"✓ find_paths: Found {len(results)} paths (limited to 20)")
    
    # Find reachable nodes from 'a'
    query = ast.queries[1]
    compiled = compiler.compile(query)
    results = list(compiled.execute_all())
    reachable = {r["X"].value for r in results}
    # From 'a' can reach: b, c, d, e, f
    assert reachable == {"b", "c", "d", "e", "f"}
    print(f"✓ find_reachable: Found {len(results)} reachable nodes from 'a'")
    
    print("✅ Complex rules test passed!\n")


if __name__ == "__main__":
    tests = [
        test_family_tree,
        test_employee_database,
        test_with_adapters,
        test_complex_rules,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Integration Tests: {passed}/{len(tests)} passed")
    print(f"{'='*50}")
    
    sys.exit(0 if failed == 0 else 1)



