"""
Comprehensive unit tests for backtracking search using correct LogicEngine API.
"""

from namel3ss.ast.logic import LogicVar, LogicAtom, LogicNumber, LogicStruct, LogicFact, LogicRule, LogicList
from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngine, LogicEngineConfig


def test_simple_fact_query():
    """Query a simple fact."""
    engine = LogicEngine()
    facts = [
        LogicFact(head=LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicAtom(value="pizza")])),
        LogicFact(head=LogicStruct(functor="likes", args=[LogicAtom(value="bob"), LogicAtom(value="burgers")]))
    ]
    
    goal = LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicVar(name="X")])
    solutions = list(engine.solve([goal], facts, []))
    
    assert len(solutions) == 1
    assert solutions[0].lookup("X") == LogicAtom(value="pizza")
    print("✓ Simple fact query works")


def test_multiple_solutions():
    """Query should return multiple solutions."""
    engine = LogicEngine()
    facts = [
        LogicFact(head=LogicStruct(functor="color", args=[LogicAtom(value="red")])),
        LogicFact(head=LogicStruct(functor="color", args=[LogicAtom(value="green")])),
        LogicFact(head=LogicStruct(functor="color", args=[LogicAtom(value="blue")]))
    ]
    
    goal = LogicStruct(functor="color", args=[LogicVar(name="X")])
    solutions = list(engine.solve([goal], facts, []))
    
    assert len(solutions) == 3
    colors = {sol.lookup("X").value for sol in solutions}
    assert colors == {"red", "green", "blue"}
    print("✓ Multiple solutions work")


def test_no_solutions():
    """Query with no matching facts should return empty."""
    engine = LogicEngine()
    facts = [
        LogicFact(head=LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicAtom(value="pizza")]))
    ]
    
    goal = LogicStruct(functor="likes", args=[LogicAtom(value="bob"), LogicVar(name="X")])
    solutions = list(engine.solve([goal], facts, []))
    
    assert len(solutions) == 0
    print("✓ No solutions handled correctly")


def test_simple_rule():
    """Query using a simple rule."""
    engine = LogicEngine()
    
    facts = [
        LogicFact(head=LogicStruct(functor="parent", args=[LogicAtom(value="alice"), LogicAtom(value="bob")])),
        LogicFact(head=LogicStruct(functor="parent", args=[LogicAtom(value="bob"), LogicAtom(value="charlie")]))
    ]
    
    # Rule: ancestor(X, Y) :- parent(X, Y).
    rules = [
        LogicRule(
            head=LogicStruct(functor="ancestor", args=[LogicVar(name="X"), LogicVar(name="Y")]),
            body=[LogicStruct(functor="parent", args=[LogicVar(name="X"), LogicVar(name="Y")])]
        )
    ]
    
    goal = LogicStruct(functor="ancestor", args=[LogicAtom(value="alice"), LogicAtom(value="bob")])
    solutions = list(engine.solve([goal], facts, rules))
    
    assert len(solutions) == 1
    print("✓ Simple rule works")


def test_recursive_rule():
    """Query using recursive rule."""
    engine = LogicEngine()
    
    facts = [
        LogicFact(head=LogicStruct(functor="parent", args=[LogicAtom(value="alice"), LogicAtom(value="bob")])),
        LogicFact(head=LogicStruct(functor="parent", args=[LogicAtom(value="bob"), LogicAtom(value="charlie")])),
        LogicFact(head=LogicStruct(functor="parent", args=[LogicAtom(value="charlie"), LogicAtom(value="dave")]))
    ]
    
    rules = [
        # ancestor(X, Y) :- parent(X, Y).
        LogicRule(
            head=LogicStruct(functor="ancestor", args=[LogicVar(name="X"), LogicVar(name="Y")]),
            body=[LogicStruct(functor="parent", args=[LogicVar(name="X"), LogicVar(name="Y")])]
        ),
        # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        LogicRule(
            head=LogicStruct(functor="ancestor", args=[LogicVar(name="X"), LogicVar(name="Z")]),
            body=[
                LogicStruct(functor="parent", args=[LogicVar(name="X"), LogicVar(name="Y")]),
                LogicStruct(functor="ancestor", args=[LogicVar(name="Y"), LogicVar(name="Z")])
            ]
        )
    ]
    
    goal = LogicStruct(functor="ancestor", args=[LogicAtom(value="alice"), LogicVar(name="X")])
    solutions = list(engine.solve([goal], facts, rules))
    
    # Should find: bob (direct), charlie (through bob), dave (through bob and charlie)
    assert len(solutions) == 3
    descendants = {sol.apply(LogicVar(name="X")).value for sol in solutions}
    assert descendants == {"bob", "charlie", "dave"}
    print("✓ Recursive rule works")


def test_conjunction():
    """Conjunction where both goals succeed."""
    engine = LogicEngine()
    
    facts = [
        LogicFact(head=LogicStruct(functor="person", args=[LogicAtom(value="alice")])),
        LogicFact(head=LogicStruct(functor="age", args=[LogicAtom(value="alice"), LogicNumber(value=30)])),
        LogicFact(head=LogicStruct(functor="person", args=[LogicAtom(value="bob")])),
        LogicFact(head=LogicStruct(functor="age", args=[LogicAtom(value="bob"), LogicNumber(value=25)]))
    ]
    
    rules = [
        # adult(X) :- person(X), age(X, Age).
        LogicRule(
            head=LogicStruct(functor="adult", args=[LogicVar(name="X")]),
            body=[
                LogicStruct(functor="person", args=[LogicVar(name="X")]),
                LogicStruct(functor="age", args=[LogicVar(name="X"), LogicVar(name="Age")])
            ]
        )
    ]
    
    goal = LogicStruct(functor="adult", args=[LogicAtom(value="alice")])
    solutions = list(engine.solve([goal], facts, rules))
    
    assert len(solutions) == 1
    # Age is bound (renamed variable), just check a solution was found
    print("✓ Conjunction works")


def test_backtracking_through_multiple_choices():
    """Backtracking should explore all choice points."""
    engine = LogicEngine()
    
    facts = [
        LogicFact(head=LogicStruct(functor="color", args=[LogicAtom(value="red")])),
        LogicFact(head=LogicStruct(functor="color", args=[LogicAtom(value="blue")])),
        LogicFact(head=LogicStruct(functor="size", args=[LogicAtom(value="small")])),
        LogicFact(head=LogicStruct(functor="size", args=[LogicAtom(value="large")]))
    ]
    
    rules = [
        # item(C, S) :- color(C), size(S).
        LogicRule(
            head=LogicStruct(functor="item", args=[LogicVar(name="C"), LogicVar(name="S")]),
            body=[
                LogicStruct(functor="color", args=[LogicVar(name="C")]),
                LogicStruct(functor="size", args=[LogicVar(name="S")])
            ]
        )
    ]
    
    goal = LogicStruct(functor="item", args=[LogicVar(name="C"), LogicVar(name="S")])
    solutions = list(engine.solve([goal], facts, rules))
    
    # 2 colors × 2 sizes = 4 combinations
    assert len(solutions) == 4
    combos = {(sol.apply(LogicVar(name="C")).value, sol.apply(LogicVar(name="S")).value) for sol in solutions}
    assert combos == {
        ("red", "small"), ("red", "large"),
        ("blue", "small"), ("blue", "large")
    }
    print("✓ Backtracking through multiple choices works")


def test_max_depth_limit():
    """Should stop when max depth exceeded."""
    from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngineDepthLimit
    
    config = LogicEngineConfig(max_depth=5)
    engine = LogicEngine(config=config)
    
    rules = [
        # Infinite recursion: loop(X) :- loop(X).
        LogicRule(
            head=LogicStruct(functor="loop", args=[LogicVar(name="X")]),
            body=[LogicStruct(functor="loop", args=[LogicVar(name="X")])]
        )
    ]
    
    goal = LogicStruct(functor="loop", args=[LogicAtom(value="test")])
    
    # Should raise depth limit exception
    try:
        solutions = list(engine.solve([goal], [], rules))
        assert False, "Should have raised LogicEngineDepthLimit"
    except LogicEngineDepthLimit:
        pass  # Expected
    
    print("✓ Max depth limit works")


def test_ground_query():
    """Ground query (no variables) should succeed or fail."""
    engine = LogicEngine()
    facts = [
        LogicFact(head=LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicAtom(value="pizza")]))
    ]
    
    # Should succeed
    goal = LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicAtom(value="pizza")])
    solutions = list(engine.solve([goal], facts, []))
    assert len(solutions) == 1
    assert len(solutions[0].bindings) == 0  # No variable bindings
    
    # Should fail
    goal = LogicStruct(functor="likes", args=[LogicAtom(value="alice"), LogicAtom(value="burgers")])
    solutions = list(engine.solve([goal], facts, []))
    assert len(solutions) == 0
    print("✓ Ground query works")


def test_nested_structures():
    """Should handle nested structures correctly."""
    engine = LogicEngine()
    
    facts = [
        # Fact: located_at(office, building("main", floor(3)))
        LogicFact(head=LogicStruct(functor="located_at", args=[
            LogicAtom(value="office"),
            LogicStruct(functor="building", args=[
                LogicAtom(value="main"),
                LogicStruct(functor="floor", args=[LogicNumber(value=3)])
            ])
        ]))
    ]
    
    # Query: located_at(office, building(Name, floor(F)))
    goal = LogicStruct(functor="located_at", args=[
        LogicAtom(value="office"),
        LogicStruct(functor="building", args=[
            LogicVar(name="Name"),
            LogicStruct(functor="floor", args=[LogicVar(name="F")])
        ])
    ])
    
    solutions = list(engine.solve([goal], facts, []))
    assert len(solutions) == 1
    assert solutions[0].lookup("Name") == LogicAtom(value="main")
    assert solutions[0].lookup("F") == LogicNumber(value=3)
    print("✓ Nested structures work")


def test_list_unification():
    """Should unify lists correctly."""
    engine = LogicEngine()
    
    facts = [
        # Fact: group([alice, bob, charlie])
        LogicFact(head=LogicStruct(functor="group", args=[
            LogicList(elements=[LogicAtom(value="alice"), LogicAtom(value="bob"), LogicAtom(value="charlie")])
        ]))
    ]
    
    # Query: group(Members)
    goal = LogicStruct(functor="group", args=[LogicVar(name="Members")])
    solutions = list(engine.solve([goal], facts, []))
    
    assert len(solutions) == 1
    members = solutions[0].lookup("Members")
    assert isinstance(members, LogicList)
    assert len(members.elements) == 3
    print("✓ List unification works")


if __name__ == "__main__":
    import sys
    tests = [
        test_simple_fact_query,
        test_multiple_solutions,
        test_no_solutions,
        test_simple_rule,
        test_recursive_rule,
        test_conjunction,
        test_backtracking_through_multiple_choices,
        test_max_depth_limit,
        test_ground_query,
        test_nested_structures,
        test_list_unification,
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
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
