"""
Simplified comprehensive unit tests for unification that match actual API.
"""

from namel3ss.ast.logic import LogicVar, LogicAtom, LogicNumber, LogicString, LogicStruct, LogicList
from namel3ss.codegen.backend.core.runtime.logic_engine import Substitution, unify


def test_unify_atoms():
    """Test atom unification."""
    # Same atoms should unify
    result = unify(LogicAtom(value="foo"), LogicAtom(value="foo"), Substitution())
    assert result is not None
    
    # Different atoms should not unify
    result = unify(LogicAtom(value="foo"), LogicAtom(value="bar"), Substitution())
    assert result is None
    print("✓ Atom unification works")


def test_unify_variables():
    """Test variable unification."""
    x = LogicVar(name="X")
    y = LogicVar(name="Y")
    a = LogicAtom(value="alice")
    
    # Variable with atom
    result = unify(x, a, Substitution())
    assert result is not None
    assert result.lookup("X") == a
    
    # Two variables
    result = unify(x, y, Substitution())
    assert result is not None
    assert len(result.bindings) == 1
    
    print("✓ Variable unification works")


def test_unify_structures():
    """Test structure unification."""
    # Same structures
    s1 = LogicStruct(functor="person", args=[LogicAtom(value="alice"), LogicNumber(value=30)])
    s2 = LogicStruct(functor="person", args=[LogicAtom(value="alice"), LogicNumber(value=30)])
    result = unify(s1, s2, Substitution())
    assert result is not None
    
    # Different functors
    s1 = LogicStruct(functor="person", args=[LogicAtom(value="alice")])
    s2 = LogicStruct(functor="employee", args=[LogicAtom(value="alice")])
    result = unify(s1, s2, Substitution())
    assert result is None
    
    # Different arity
    s1 = LogicStruct(functor="person", args=[LogicAtom(value="alice")])
    s2 = LogicStruct(functor="person", args=[LogicAtom(value="alice"), LogicNumber(value=30)])
    result = unify(s1, s2, Substitution())
    assert result is None
    
    print("✓ Structure unification works")


def test_unify_structures_with_variables():
    """Test structure unification with variables."""
    s1 = LogicStruct(functor="person", args=[LogicVar(name="X"), LogicNumber(value=30)])
    s2 = LogicStruct(functor="person", args=[LogicAtom(value="alice"), LogicVar(name="Y")])
    
    result = unify(s1, s2, Substitution())
    assert result is not None
    assert result.lookup("X") == LogicAtom(value="alice")
    assert result.lookup("Y") == LogicNumber(value=30)
    
    print("✓ Structure unification with variables works")


def test_occurs_check():
    """Test occurs check prevents infinite structures."""
    x = LogicVar(name="X")
    # X = f(X) should fail
    t = LogicStruct(functor="f", args=[x])
    result = unify(x, t, Substitution())
    assert result is None
    
    # X = f(g(h(X))) should fail
    t = LogicStruct(functor="f", args=[
        LogicStruct(functor="g", args=[
            LogicStruct(functor="h", args=[x])
        ])
    ])
    result = unify(x, t, Substitution())
    assert result is None
    
    # X = [a, X, b] should fail
    t = LogicList(elements=[LogicAtom(value="a"), x, LogicAtom(value="b")])
    result = unify(x, t, Substitution())
    assert result is None
    
    print("✓ Occurs check works")


def test_list_unification():
    """Test list unification."""
    # Same lists
    l1 = LogicList(elements=[LogicAtom(value="a"), LogicAtom(value="b")])
    l2 = LogicList(elements=[LogicAtom(value="a"), LogicAtom(value="b")])
    result = unify(l1, l2, Substitution())
    assert result is not None
    
    # Different lengths
    l1 = LogicList(elements=[LogicAtom(value="a")])
    l2 = LogicList(elements=[LogicAtom(value="a"), LogicAtom(value="b")])
    result = unify(l1, l2, Substitution())
    assert result is None
    
    # With variables
    l1 = LogicList(elements=[LogicVar(name="X"), LogicAtom(value="b")])
    l2 = LogicList(elements=[LogicAtom(value="a"), LogicVar(name="Y")])
    result = unify(l1, l2, Substitution())
    assert result is not None
    assert result.lookup("X") == LogicAtom(value="a")
    assert result.lookup("Y") == LogicAtom(value="b")
    
    print("✓ List unification works")


def test_substitution_apply():
    """Test substitution apply follows chains."""
    x = LogicVar(name="X")
    y = LogicVar(name="Y")
    z = LogicVar(name="Z")
    a = LogicAtom(value="final")
    
    sub = Substitution()
    sub.bindings["X"] = y
    sub.bindings["Y"] = z
    sub.bindings["Z"] = a
    
    # Apply should follow X -> Y -> Z -> "final"
    result = sub.apply(x)
    assert result == a
    
    print("✓ Substitution apply works")


def test_complex_unification():
    """Test complex nested unification."""
    # Build: parent(person(alice, 30), person(Child, Age))
    t1 = LogicStruct(functor="parent", args=[
        LogicStruct(functor="person", args=[LogicAtom(value="alice"), LogicNumber(value=30)]),
        LogicStruct(functor="person", args=[LogicVar(name="Child"), LogicVar(name="Age")])
    ])
    
    # Build: parent(person(Parent, ParentAge), person(bob, 5))
    t2 = LogicStruct(functor="parent", args=[
        LogicStruct(functor="person", args=[LogicVar(name="Parent"), LogicVar(name="ParentAge")]),
        LogicStruct(functor="person", args=[LogicAtom(value="bob"), LogicNumber(value=5)])
    ])
    
    result = unify(t1, t2, Substitution())
    assert result is not None
    assert result.lookup("Parent") == LogicAtom(value="alice")
    assert result.lookup("ParentAge") == LogicNumber(value=30)
    assert result.lookup("Child") == LogicAtom(value="bob")
    assert result.lookup("Age") == LogicNumber(value=5)
    
    print("✓ Complex nested unification works")


def test_unification_with_partial_bindings():
    """Test unification respects existing bindings."""
    x = LogicVar(name="X")
    y = LogicVar(name="Y")
    
    # Pre-bind X to "alice"
    sub = Substitution()
    sub.bindings["X"] = LogicAtom(value="alice")
    
    # Unify person(X, 30) with person(Y, 30)
    t1 = LogicStruct(functor="person", args=[x, LogicNumber(value=30)])
    t2 = LogicStruct(functor="person", args=[y, LogicNumber(value=30)])
    
    result = unify(t1, t2, sub)
    assert result is not None
    # Y should be bound to "alice" (through X)
    assert result.apply(y) == LogicAtom(value="alice")
    
    print("✓ Unification with partial bindings works")


if __name__ == "__main__":
    import sys
    tests = [
        test_unify_atoms,
        test_unify_variables,
        test_unify_structures,
        test_unify_structures_with_variables,
        test_occurs_check,
        test_list_unification,
        test_substitution_apply,
        test_complex_unification,
        test_unification_with_partial_bindings,
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


