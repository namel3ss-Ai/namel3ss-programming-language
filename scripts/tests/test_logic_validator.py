#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test logic validator."""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from namel3ss.ast.logic import (
    KnowledgeModule,
    LogicAtom,
    LogicFact,
    LogicQuery,
    LogicRule,
    LogicStruct,
    LogicVar,
)
from namel3ss.logic_validator import (
    LogicValidator,
    validate_logic_constructs,
)


def test_arity_consistency():
    """Test detection of arity inconsistencies."""
    print("Testing arity consistency validation...")
    
    # Create facts with inconsistent arity
    module = KnowledgeModule(
        name="test",
        facts=[
            LogicFact(
                head=LogicStruct(
                    functor="likes",
                    args=[LogicAtom(value="alice"), LogicAtom(value="bob")],
                    line=1
                ),
                line=1
            ),
            LogicFact(
                head=LogicStruct(
                    functor="likes",
                    args=[
                        LogicAtom(value="charlie"),
                        LogicAtom(value="cake"),
                        LogicAtom(value="very_much")
                    ],
                    line=2
                ),
                line=2
            ),
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    assert len(errors) > 0, "Expected arity mismatch error"
    assert any("inconsistent arities" in err for err in errors)
    print(f"  ✓ Detected arity mismatch: {errors[0]}")
    print()


def test_undefined_module():
    """Test detection of undefined knowledge module references."""
    print("Testing undefined module detection...")
    
    # Create query referencing non-existent module
    query = LogicQuery(
        name="test_query",
        knowledge_sources=["nonexistent_module"],
        goals=[LogicStruct(functor="test", args=[])],
    )
    
    errors, warnings = validate_logic_constructs([], [query])
    
    assert len(errors) > 0, "Expected undefined module error"
    assert any("undefined knowledge module" in err for err in errors)
    print(f"  ✓ Detected undefined module: {errors[0]}")
    print()


def test_unsafe_variables():
    """Test detection of unsafe variables in rule heads."""
    print("Testing unsafe variable detection...")
    
    # Create rule with variable in head but not in body
    module = KnowledgeModule(
        name="test",
        rules=[
            LogicRule(
                head=LogicStruct(
                    functor="bad_rule",
                    args=[LogicVar(name="X"), LogicVar(name="Y")],
                    line=1
                ),
                body=[
                    LogicStruct(
                        functor="something",
                        args=[LogicVar(name="X")],
                        line=1
                    )
                ],
                line=1
            )
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    assert len(errors) > 0, "Expected unsafe variable error"
    assert any("unsafe variables" in err and "Y" in err for err in errors)
    print(f"  ✓ Detected unsafe variable: {errors[0]}")
    print()


def test_singleton_variables():
    """Test detection of singleton variables (potential typos)."""
    print("Testing singleton variable detection...")
    
    # Create rule with singleton variable
    module = KnowledgeModule(
        name="test",
        rules=[
            LogicRule(
                head=LogicStruct(
                    functor="has_typo",
                    args=[LogicVar(name="X")],
                    line=1
                ),
                body=[
                    LogicStruct(
                        functor="something",
                        args=[LogicVar(name="X")],
                        line=1
                    ),
                    LogicStruct(
                        functor="other",
                        args=[LogicVar(name="Y")],  # Y appears only once - likely typo
                        line=1
                    )
                ],
                line=1
            )
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    assert len(warnings) > 0, "Expected singleton variable warning"
    assert any("singleton variables" in warn and "Y" in warn for warn in warnings)
    print(f"  ✓ Detected singleton variable: {warnings[0]}")
    print()


def test_variables_in_facts():
    """Test warning for variables in facts (should be ground)."""
    print("Testing variables in facts detection...")
    
    # Create fact with variables
    module = KnowledgeModule(
        name="test",
        facts=[
            LogicFact(
                head=LogicStruct(
                    functor="bad_fact",
                    args=[LogicVar(name="X"), LogicAtom(value="value")],
                    line=1
                ),
                line=1
            )
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    assert len(warnings) > 0, "Expected variable in fact warning"
    assert any("contains variables" in warn for warn in warnings)
    print(f"  ✓ Detected variable in fact: {warnings[0]}")
    print()


def test_undefined_predicate():
    """Test warning for predicates used but never defined."""
    print("Testing undefined predicate detection...")
    
    # Create rule using undefined predicate
    module = KnowledgeModule(
        name="test",
        rules=[
            LogicRule(
                head=LogicStruct(
                    functor="defined",
                    args=[LogicVar(name="X")],
                    line=1
                ),
                body=[
                    LogicStruct(
                        functor="undefined_pred",  # Never defined
                        args=[LogicVar(name="X")],
                        line=1
                    )
                ],
                line=1
            )
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    assert len(warnings) > 0, "Expected undefined predicate warning"
    assert any("used but never defined" in warn for warn in warnings)
    print(f"  ✓ Detected undefined predicate: {warnings[0]}")
    print()


def test_valid_module():
    """Test that valid modules pass validation."""
    print("Testing valid module...")
    
    # Create valid module
    module = KnowledgeModule(
        name="test",
        facts=[
            LogicFact(
                head=LogicStruct(
                    functor="parent",
                    args=[LogicAtom(value="alice"), LogicAtom(value="bob")],
                )
            ),
        ],
        rules=[
            LogicRule(
                head=LogicStruct(
                    functor="ancestor",
                    args=[LogicVar(name="X"), LogicVar(name="Y")],
                ),
                body=[
                    LogicStruct(
                        functor="parent",
                        args=[LogicVar(name="X"), LogicVar(name="Y")],
                    )
                ]
            )
        ]
    )
    
    errors, warnings = validate_logic_constructs([module], [])
    
    # Should have only undefined predicate warning (ancestor used but not defined as fact)
    # But no errors
    assert len(errors) == 0, f"Expected no errors, got: {errors}"
    print(f"  ✓ Valid module passed validation")
    if warnings:
        print(f"  ℹ Warnings: {warnings}")
    print()


def test_query_projected_variable():
    """Test warning for projected variables not in goals."""
    print("Testing query projected variable validation...")
    
    module = KnowledgeModule(name="test")
    
    query = LogicQuery(
        name="test_query",
        knowledge_sources=["test"],
        goals=[
            LogicStruct(
                functor="test",
                args=[LogicVar(name="X")],
            )
        ],
        variables=["X", "Y"]  # Y not in goals
    )
    
    errors, warnings = validate_logic_constructs([module], [query])
    
    assert len(warnings) > 0, "Expected warning for unused projected variable"
    assert any("doesn't appear in any goal" in warn and "Y" in warn for warn in warnings)
    print(f"  ✓ Detected unused projected variable: {warnings[0]}")
    print()


if __name__ == '__main__':
    test_arity_consistency()
    test_undefined_module()
    test_unsafe_variables()
    test_singleton_variables()
    test_variables_in_facts()
    test_undefined_predicate()
    test_valid_module()
    test_query_projected_variable()
    
    print("=" * 60)
    print("All logic validator tests passed!")
