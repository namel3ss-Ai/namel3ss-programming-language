#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the logic engine with unification and backtracking."""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from namel3ss.ast.logic import (
    LogicAtom,
    LogicFact,
    LogicNumber,
    LogicRule,
    LogicStruct,
    LogicVar,
)
from namel3ss.codegen.backend.core.runtime.logic_engine import (
    LogicEngine,
    Substitution,
    unify,
)


def test_unification():
    """Test basic unification."""
    print("Testing unification...")
    
    # Test 1: Unify X with atom
    x = LogicVar(name="X")
    alice = LogicAtom(value="alice")
    subst = unify(x, alice, Substitution())
    assert subst is not None
    assert subst.lookup("X") == alice
    print(f"  ✓ X unifies with alice: {subst}")
    
    # Test 2: Unify structures
    parent1 = LogicStruct(functor="parent", args=[alice, LogicVar(name="Y")])
    bob = LogicAtom(value="bob")
    parent2 = LogicStruct(functor="parent", args=[alice, bob])
    subst = unify(parent1, parent2, Substitution())
    assert subst is not None
    assert subst.lookup("Y") == bob
    print(f"  ✓ parent(alice, Y) unifies with parent(alice, bob): {subst}")
    
    # Test 3: Unification failure - different functors
    likes = LogicStruct(functor="likes", args=[alice, bob])
    subst = unify(parent2, likes, Substitution())
    assert subst is None
    print(f"  ✓ parent(...) does not unify with likes(...)")
    
    print("Unification tests passed!\n")


def test_simple_query():
    """Test simple fact queries."""
    print("Testing simple fact queries...")
    
    # Create facts: parent(alice, bob), parent(bob, charlie)
    facts = [
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="alice"),
            LogicAtom(value="bob")
        ])),
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="bob"),
            LogicAtom(value="charlie")
        ])),
    ]
    
    # Query: parent(alice, Who)?
    goal = LogicStruct(functor="parent", args=[
        LogicAtom(value="alice"),
        LogicVar(name="Who")
    ])
    
    engine = LogicEngine()
    solutions = list(engine.solve([goal], facts, []))
    
    assert len(solutions) == 1
    assert solutions[0].lookup("Who") == LogicAtom(value="bob")
    print(f"  ✓ parent(alice, Who)? => Who = {solutions[0].lookup('Who')}")
    
    # Query: parent(Who, charlie)?
    goal2 = LogicStruct(functor="parent", args=[
        LogicVar(name="Who"),
        LogicAtom(value="charlie")
    ])
    
    solutions2 = list(engine.solve([goal2], facts, []))
    assert len(solutions2) == 1
    assert solutions2[0].lookup("Who") == LogicAtom(value="bob")
    print(f"  ✓ parent(Who, charlie)? => Who = {solutions2[0].lookup('Who')}")
    
    print("Simple query tests passed!\n")


def test_recursive_rules():
    """Test recursive rules with backtracking."""
    print("Testing recursive rules...")
    
    # Facts: parent(alice, bob), parent(bob, charlie)
    facts = [
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="alice"),
            LogicAtom(value="bob")
        ])),
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="bob"),
            LogicAtom(value="charlie")
        ])),
    ]
    
    # Rules:
    # ancestor(X, Y) :- parent(X, Y).
    # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
    rules = [
        LogicRule(
            head=LogicStruct(functor="ancestor", args=[
                LogicVar(name="X"),
                LogicVar(name="Y")
            ]),
            body=[
                LogicStruct(functor="parent", args=[
                    LogicVar(name="X"),
                    LogicVar(name="Y")
                ])
            ]
        ),
        LogicRule(
            head=LogicStruct(functor="ancestor", args=[
                LogicVar(name="X"),
                LogicVar(name="Z")
            ]),
            body=[
                LogicStruct(functor="parent", args=[
                    LogicVar(name="X"),
                    LogicVar(name="Y")
                ]),
                LogicStruct(functor="ancestor", args=[
                    LogicVar(name="Y"),
                    LogicVar(name="Z")
                ])
            ]
        ),
    ]
    
    # Query: ancestor(alice, Who)?
    goal = LogicStruct(functor="ancestor", args=[
        LogicAtom(value="alice"),
        LogicVar(name="Who")
    ])
    
    engine = LogicEngine()
    solutions = list(engine.solve([goal], facts, rules))
    
    print(f"  Found {len(solutions)} solutions")
    for i, sol in enumerate(solutions):
        print(f"    Solution {i+1} bindings: {sol}")
        who_term = sol.lookup("Who")
        print(f"      Initial lookup: Who = {who_term} (type: {type(who_term).__name__})")
        # Apply substitution to resolve chains
        who_resolved = sol.apply(LogicVar(name="Who"))
        print(f"      After apply: Who = {who_resolved} (type: {type(who_resolved).__name__})")
    
    # Should find: bob (direct) and charlie (transitive)
    assert len(solutions) == 2
    
    who_values = []
    for sol in solutions:
        # Apply substitution to resolve variable chains
        who_term = sol.apply(LogicVar(name="Who"))
        if isinstance(who_term, LogicAtom):
            who_values.append(who_term.value)
    
    print(f"  Extracted values: {who_values}")
    assert "bob" in who_values
    assert "charlie" in who_values
    
    print(f"  ✓ ancestor(alice, Who)? found {len(solutions)} solutions:")
    for sol in solutions:
        print(f"    - Who = {sol.lookup('Who')}")
    
    print("Recursive rule tests passed!\n")


def test_conjunction():
    """Test conjunctive goals."""
    print("Testing conjunctive goals...")
    
    # Facts
    facts = [
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="alice"), LogicAtom(value="bob")
        ])),
        LogicFact(head=LogicStruct(functor="parent", args=[
            LogicAtom(value="bob"), LogicAtom(value="charlie")
        ])),
        LogicFact(head=LogicStruct(functor="age", args=[
            LogicAtom(value="bob"), LogicNumber(value=30)
        ])),
        LogicFact(head=LogicStruct(functor="age", args=[
            LogicAtom(value="charlie"), LogicNumber(value=10)
        ])),
    ]
    
    # Query: parent(alice, X), age(X, Age)?
    goals = [
        LogicStruct(functor="parent", args=[
            LogicAtom(value="alice"),
            LogicVar(name="X")
        ]),
        LogicStruct(functor="age", args=[
            LogicVar(name="X"),
            LogicVar(name="Age")
        ]),
    ]
    
    engine = LogicEngine()
    solutions = list(engine.solve(goals, facts, []))
    
    assert len(solutions) == 1
    assert solutions[0].lookup("X") == LogicAtom(value="bob")
    assert solutions[0].lookup("Age") == LogicNumber(value=30)
    print(f"  ✓ parent(alice, X), age(X, Age)? => X = {solutions[0].lookup('X')}, Age = {solutions[0].lookup('Age')}")
    
    print("Conjunction tests passed!\n")


if __name__ == '__main__':
    test_unification()
    test_simple_query()
    test_recursive_rules()
    test_conjunction()
    print("=" * 50)
    print("All logic engine tests passed!")
