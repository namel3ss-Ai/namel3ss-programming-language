"""
Comprehensive tests for Prolog-style list pattern matching in the logic engine.

Tests full list pattern unification including:
- [H|T] patterns
- [H1, H2|T] patterns with multiple head elements
- [H|[]] patterns (single element lists)
- Nested list patterns
- Variable bindings in list patterns
- Edge cases and failures
"""

import pytest

from namel3ss.ast.logic import (
    LogicAtom,
    LogicList,
    LogicNumber,
    LogicStruct,
    LogicVar,
)
from namel3ss.codegen.backend.core.runtime.logic_engine import Substitution, unify


# ============================================================================
# Basic List Pattern Tests
# ============================================================================

def test_empty_list_unification():
    """Test that empty lists unify."""
    list1 = LogicList(elements=[])
    list2 = LogicList(elements=[])
    
    result = unify(list1, list2, Substitution())
    assert result is not None
    print("✓ Empty lists unify")


def test_simple_list_unification():
    """Test that simple lists with same elements unify."""
    list1 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    list2 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    
    result = unify(list1, list2, Substitution())
    assert result is not None
    print("✓ Simple lists unify")


def test_simple_list_different_length_fails():
    """Test that lists with different lengths don't unify."""
    list1 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
    ])
    list2 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    
    result = unify(list1, list2, Substitution())
    assert result is None
    print("✓ Lists with different lengths don't unify")


# ============================================================================
# [H|T] Pattern Tests
# ============================================================================

def test_head_tail_pattern_with_empty_list_fails():
    """Test that [H|T] doesn't unify with []."""
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    empty_list = LogicList(elements=[])
    
    result = unify(pattern, empty_list, Substitution())
    assert result is None
    print("✓ [H|T] doesn't unify with []")


def test_head_tail_pattern_with_single_element():
    """Test that [H|T] unifies with [a], binding H=a and T=[]."""
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    single_list = LogicList(elements=[LogicAtom(value="a")])
    
    result = unify(pattern, single_list, Substitution())
    assert result is not None
    
    # Check H is bound to 'a'
    h_binding = result.apply(LogicVar(name="H"))
    assert isinstance(h_binding, LogicAtom)
    assert h_binding.value == "a"
    
    # Check T is bound to []
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    assert len(t_binding.elements) == 0
    assert t_binding.tail is None
    
    print("✓ [H|T] unifies with [a], H=a, T=[]")


def test_head_tail_pattern_with_multiple_elements():
    """Test that [H|T] unifies with [a,b,c], binding H=a and T=[b,c]."""
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    multi_list = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    
    result = unify(pattern, multi_list, Substitution())
    assert result is not None
    
    # Check H is bound to 'a'
    h_binding = result.apply(LogicVar(name="H"))
    assert isinstance(h_binding, LogicAtom)
    assert h_binding.value == "a"
    
    # Check T is bound to [b, c]
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    assert len(t_binding.elements) == 2
    assert t_binding.elements[0].value == "b"
    assert t_binding.elements[1].value == "c"
    assert t_binding.tail is None
    
    print("✓ [H|T] unifies with [a,b,c], H=a, T=[b,c]")


def test_head_tail_pattern_symmetric():
    """Test that unification is symmetric: [a,b,c] with [H|T]."""
    multi_list = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    
    result = unify(multi_list, pattern, Substitution())
    assert result is not None
    
    h_binding = result.apply(LogicVar(name="H"))
    assert h_binding.value == "a"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert len(t_binding.elements) == 2
    
    print("✓ Unification is symmetric")


# ============================================================================
# [H1, H2|T] Pattern Tests (Multiple Head Elements)
# ============================================================================

def test_two_heads_tail_pattern():
    """Test that [H1,H2|T] unifies with [a,b,c,d], binding H1=a, H2=b, T=[c,d]."""
    pattern = LogicList(
        elements=[LogicVar(name="H1"), LogicVar(name="H2")],
        tail=LogicVar(name="T")
    )
    list4 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
        LogicAtom(value="d"),
    ])
    
    result = unify(pattern, list4, Substitution())
    assert result is not None
    
    h1_binding = result.apply(LogicVar(name="H1"))
    assert h1_binding.value == "a"
    
    h2_binding = result.apply(LogicVar(name="H2"))
    assert h2_binding.value == "b"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    assert len(t_binding.elements) == 2
    assert t_binding.elements[0].value == "c"
    assert t_binding.elements[1].value == "d"
    
    print("✓ [H1,H2|T] unifies with [a,b,c,d], H1=a, H2=b, T=[c,d]")


def test_two_heads_tail_pattern_exact_match():
    """Test that [H1,H2|T] unifies with [a,b], binding H1=a, H2=b, T=[]."""
    pattern = LogicList(
        elements=[LogicVar(name="H1"), LogicVar(name="H2")],
        tail=LogicVar(name="T")
    )
    list2 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
    ])
    
    result = unify(pattern, list2, Substitution())
    assert result is not None
    
    h1_binding = result.apply(LogicVar(name="H1"))
    assert h1_binding.value == "a"
    
    h2_binding = result.apply(LogicVar(name="H2"))
    assert h2_binding.value == "b"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    assert len(t_binding.elements) == 0
    
    print("✓ [H1,H2|T] unifies with [a,b], H1=a, H2=b, T=[]")


def test_two_heads_tail_pattern_too_short_fails():
    """Test that [H1,H2|T] doesn't unify with [a]."""
    pattern = LogicList(
        elements=[LogicVar(name="H1"), LogicVar(name="H2")],
        tail=LogicVar(name="T")
    )
    list1 = LogicList(elements=[LogicAtom(value="a")])
    
    result = unify(pattern, list1, Substitution())
    assert result is None
    print("✓ [H1,H2|T] doesn't unify with [a]")


def test_three_heads_tail_pattern():
    """Test [H1,H2,H3|T] pattern."""
    pattern = LogicList(
        elements=[
            LogicVar(name="H1"),
            LogicVar(name="H2"),
            LogicVar(name="H3")
        ],
        tail=LogicVar(name="T")
    )
    list5 = LogicList(elements=[
        LogicNumber(value=1),
        LogicNumber(value=2),
        LogicNumber(value=3),
        LogicNumber(value=4),
        LogicNumber(value=5),
    ])
    
    result = unify(pattern, list5, Substitution())
    assert result is not None
    
    h1_binding = result.apply(LogicVar(name="H1"))
    assert h1_binding.value == 1
    
    h2_binding = result.apply(LogicVar(name="H2"))
    assert h2_binding.value == 2
    
    h3_binding = result.apply(LogicVar(name="H3"))
    assert h3_binding.value == 3
    
    t_binding = result.apply(LogicVar(name="T"))
    assert len(t_binding.elements) == 2
    assert t_binding.elements[0].value == 4
    assert t_binding.elements[1].value == 5
    
    print("✓ [H1,H2,H3|T] pattern works")


# ============================================================================
# [H|[]] Pattern Tests (Empty Tail)
# ============================================================================

def test_head_empty_tail_pattern():
    """Test that [H|[]] unifies only with single-element lists."""
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicList(elements=[])  # Explicit empty list as tail
    )
    
    # Should unify with [a]
    single = LogicList(elements=[LogicAtom(value="a")])
    result = unify(pattern, single, Substitution())
    assert result is not None
    
    h_binding = result.apply(LogicVar(name="H"))
    assert h_binding.value == "a"
    
    # Should NOT unify with [a, b]
    double = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
    ])
    result = unify(pattern, double, Substitution())
    assert result is None
    
    # Should NOT unify with []
    empty = LogicList(elements=[])
    result = unify(pattern, empty, Substitution())
    assert result is None
    
    print("✓ [H|[]] pattern works correctly")


# ============================================================================
# Pattern-Pattern Unification
# ============================================================================

def test_two_patterns_same_structure():
    """Test unifying two patterns with same structure."""
    pattern1 = LogicList(
        elements=[LogicVar(name="X")],
        tail=LogicVar(name="Xs")
    )
    pattern2 = LogicList(
        elements=[LogicVar(name="Y")],
        tail=LogicVar(name="Ys")
    )
    
    result = unify(pattern1, pattern2, Substitution())
    assert result is not None
    
    # X and Y should be unified
    x_binding = result.apply(LogicVar(name="X"))
    y_binding = result.apply(LogicVar(name="Y"))
    
    # They should unify to the same variable or one binds to the other
    # Check that applying substitution makes them consistent
    assert (isinstance(x_binding, LogicVar) and x_binding.name == "Y") or \
           (isinstance(y_binding, LogicVar) and y_binding.name == "X")
    
    print("✓ Two patterns with same structure unify")


def test_two_patterns_different_head_count():
    """Test unifying [H|T] with [X1,X2|Xs]."""
    pattern1 = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    pattern2 = LogicList(
        elements=[LogicVar(name="X1"), LogicVar(name="X2")],
        tail=LogicVar(name="Xs")
    )
    
    result = unify(pattern1, pattern2, Substitution())
    assert result is not None
    
    # H should unify with X1
    h_binding = result.apply(LogicVar(name="H"))
    x1_binding = result.apply(LogicVar(name="X1"))
    
    # T should unify with [X2|Xs]
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    
    print("✓ Patterns with different head counts unify correctly")


# ============================================================================
# Nested Patterns
# ============================================================================

def test_nested_list_in_structure():
    """Test structure containing list pattern: foo([H|T])."""
    pattern = LogicStruct(
        functor="foo",
        args=[
            LogicList(
                elements=[LogicVar(name="H")],
                tail=LogicVar(name="T")
            )
        ]
    )
    
    concrete = LogicStruct(
        functor="foo",
        args=[
            LogicList(elements=[
                LogicAtom(value="a"),
                LogicAtom(value="b"),
            ])
        ]
    )
    
    result = unify(pattern, concrete, Substitution())
    assert result is not None
    
    h_binding = result.apply(LogicVar(name="H"))
    assert h_binding.value == "a"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert len(t_binding.elements) == 1
    assert t_binding.elements[0].value == "b"
    
    print("✓ Nested list patterns in structures work")


def test_list_of_lists_pattern():
    """Test pattern matching on list of lists: [[H|T]|Rest]."""
    inner_pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    outer_pattern = LogicList(
        elements=[inner_pattern],
        tail=LogicVar(name="Rest")
    )
    
    inner_list1 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
    ])
    inner_list2 = LogicList(elements=[
        LogicAtom(value="c"),
    ])
    concrete = LogicList(elements=[inner_list1, inner_list2])
    
    result = unify(outer_pattern, concrete, Substitution())
    assert result is not None
    
    h_binding = result.apply(LogicVar(name="H"))
    assert h_binding.value == "a"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert len(t_binding.elements) == 1
    
    rest_binding = result.apply(LogicVar(name="Rest"))
    assert len(rest_binding.elements) == 1
    
    print("✓ Nested list patterns work")


# ============================================================================
# Variable Sharing and Consistency
# ============================================================================

def test_same_variable_in_pattern():
    """Test that using same variable twice enforces equality."""
    # Pattern: [X, X] - both elements must be the same
    pattern = LogicList(elements=[
        LogicVar(name="X"),
        LogicVar(name="X"),
    ])
    
    # Should unify with [a, a]
    same = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="a"),
    ])
    result = unify(pattern, same, Substitution())
    assert result is not None
    
    # Should NOT unify with [a, b]
    different = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
    ])
    result = unify(pattern, different, Substitution())
    assert result is None
    
    print("✓ Same variable in pattern enforces equality")


def test_same_variable_in_head_and_tail():
    """Test pattern [X|X] - head and tail must be compatible."""
    pattern = LogicList(
        elements=[LogicVar(name="X")],
        tail=LogicVar(name="X")
    )
    
    # This creates a constraint: X must equal [X|X], which is impossible
    # (infinite structure), so this should fail the occurs check
    single = LogicList(elements=[LogicAtom(value="a")])
    result = unify(pattern, single, Substitution())
    
    # The occurs check should prevent this
    assert result is None
    
    print("✓ Occurs check prevents [X|X] pattern")


# ============================================================================
# Concrete Value Patterns
# ============================================================================

def test_pattern_with_concrete_head():
    """Test pattern with concrete values in head: [a, X, c|T]."""
    pattern = LogicList(
        elements=[
            LogicAtom(value="a"),
            LogicVar(name="X"),
            LogicAtom(value="c"),
        ],
        tail=LogicVar(name="T")
    )
    
    # Should match [a, b, c, d]
    list4 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
        LogicAtom(value="d"),
    ])
    
    result = unify(pattern, list4, Substitution())
    assert result is not None
    
    x_binding = result.apply(LogicVar(name="X"))
    assert x_binding.value == "b"
    
    t_binding = result.apply(LogicVar(name="T"))
    assert len(t_binding.elements) == 1
    assert t_binding.elements[0].value == "d"
    
    # Should NOT match [a, b, x, d] (third element must be 'c')
    list_wrong = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="x"),
        LogicAtom(value="d"),
    ])
    result = unify(pattern, list_wrong, Substitution())
    assert result is None
    
    print("✓ Patterns with concrete head values work")


# ============================================================================
# Edge Cases
# ============================================================================

def test_tail_bound_to_specific_list():
    """Test that tail can be bound to a specific list value."""
    pattern = LogicList(
        elements=[LogicVar(name="H")],
        tail=LogicVar(name="T")
    )
    
    # Unify with [a,b,c]
    list3 = LogicList(elements=[
        LogicAtom(value="a"),
        LogicAtom(value="b"),
        LogicAtom(value="c"),
    ])
    
    result = unify(pattern, list3, Substitution())
    assert result is not None
    
    # Now use the result to unify another pattern with specific tail
    pattern2 = LogicList(
        elements=[LogicVar(name="Y")],
        tail=LogicList(elements=[
            LogicAtom(value="b"),
            LogicAtom(value="c"),
        ])
    )
    
    result2 = unify(pattern2, list3, result)
    assert result2 is not None
    
    y_binding = result2.apply(LogicVar(name="Y"))
    assert y_binding.value == "a"
    
    print("✓ Tail can be bound to specific list")


def test_empty_list_with_empty_tail_variable():
    """Test [] with [|T] where T must be []."""
    empty = LogicList(elements=[])
    pattern = LogicList(elements=[], tail=LogicVar(name="T"))
    
    result = unify(empty, pattern, Substitution())
    assert result is not None
    
    t_binding = result.apply(LogicVar(name="T"))
    assert isinstance(t_binding, LogicList)
    assert len(t_binding.elements) == 0
    
    print("✓ Empty list with tail variable works")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    test_empty_list_unification()
    test_simple_list_unification()
    test_simple_list_different_length_fails()
    
    test_head_tail_pattern_with_empty_list_fails()
    test_head_tail_pattern_with_single_element()
    test_head_tail_pattern_with_multiple_elements()
    test_head_tail_pattern_symmetric()
    
    test_two_heads_tail_pattern()
    test_two_heads_tail_pattern_exact_match()
    test_two_heads_tail_pattern_too_short_fails()
    test_three_heads_tail_pattern()
    
    test_head_empty_tail_pattern()
    
    test_two_patterns_same_structure()
    test_two_patterns_different_head_count()
    
    test_nested_list_in_structure()
    test_list_of_lists_pattern()
    
    test_same_variable_in_pattern()
    test_same_variable_in_head_and_tail()
    
    test_pattern_with_concrete_head()
    
    test_tail_bound_to_specific_list()
    test_empty_list_with_empty_tail_variable()
    
    print("\n" + "="*60)
    print("✓ ALL LIST PATTERN TESTS PASSED")
    print("="*60)
