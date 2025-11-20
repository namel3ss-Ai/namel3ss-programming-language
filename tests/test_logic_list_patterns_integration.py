"""
Integration tests for list pattern matching in the full query compiler and engine.

Tests end-to-end list pattern matching through:
- Knowledge base rules with list patterns
- Query compilation
- Query execution
- Variable binding extraction
"""

import pytest

from namel3ss.ast.logic import (
    KnowledgeModule,
    LogicAtom,
    LogicFact,
    LogicList,
    LogicNumber,
    LogicQuery,
    LogicRule,
    LogicStruct,
    LogicVar,
)
from namel3ss.codegen.backend.core.runtime.query_compiler import (
    QueryCompiler,
    QueryContext,
)
from namel3ss.codegen.backend.core.runtime.logic_engine import (
    LogicEngine,
    LogicEngineConfig,
    Substitution,
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_module(name: str, facts: list, rules: list) -> KnowledgeModule:
    """Helper to create a knowledge module for testing."""
    return KnowledgeModule(
        name=name,
        facts=facts,
        rules=rules,
    )


# ============================================================================
# List Processing Rules Tests
# ============================================================================

def test_list_head_extraction():
    """Test rule that extracts head of a list."""
    # Rule: head([H|_], H).
    rule = LogicRule(
        head=LogicStruct(
            functor="head",
            args=[
                LogicList(
                    elements=[LogicVar(name="H")],
                    tail=LogicVar(name="_")
                ),
                LogicVar(name="H")
            ]
        ),
        body=[]
    )
    
    # Fact: No facts needed
    module = create_test_module("lists", facts=[], rules=[rule])
    
    # Query: head([1,2,3], X)?
    query_struct = LogicStruct(
        functor="head",
        args=[
            LogicList(elements=[
                LogicNumber(value=1),
                LogicNumber(value=2),
                LogicNumber(value=3),
            ]),
            LogicVar(name="X")
        ]
    )
    
    # Execute
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], [rule]))
    
    assert len(solutions) == 1
    x_value = solutions[0].apply(LogicVar(name="X"))
    assert isinstance(x_value, LogicNumber)
    assert x_value.value == 1
    
    print("✓ List head extraction works")


def test_list_tail_extraction():
    """Test rule that extracts tail of a list."""
    # Rule: tail([_|T], T).
    rule = LogicRule(
        head=LogicStruct(
            functor="tail",
            args=[
                LogicList(
                    elements=[LogicVar(name="_")],
                    tail=LogicVar(name="T")
                ),
                LogicVar(name="T")
            ]
        ),
        body=[]
    )
    
    module = create_test_module("lists", facts=[], rules=[rule])
    
    # Query: tail([a,b,c], X)?
    query_struct = LogicStruct(
        functor="tail",
        args=[
            LogicList(elements=[
                LogicAtom(value="a"),
                LogicAtom(value="b"),
                LogicAtom(value="c"),
            ]),
            LogicVar(name="X")
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], [rule]))
    
    assert len(solutions) == 1
    x_value = solutions[0].apply(LogicVar(name="X"))
    assert isinstance(x_value, LogicList)
    assert len(x_value.elements) == 2
    assert x_value.elements[0].value == "b"
    assert x_value.elements[1].value == "c"
    
    print("✓ List tail extraction works")


def test_list_member_check():
    """Test recursive member/2 predicate."""
    # Rule 1: member(X, [X|_]).
    rule1 = LogicRule(
        head=LogicStruct(
            functor="member",
            args=[
                LogicVar(name="X"),
                LogicList(
                    elements=[LogicVar(name="X")],
                    tail=LogicVar(name="_")
                )
            ]
        ),
        body=[]
    )
    
    # Rule 2: member(X, [_|T]) :- member(X, T).
    rule2 = LogicRule(
        head=LogicStruct(
            functor="member",
            args=[
                LogicVar(name="X"),
                LogicList(
                    elements=[LogicVar(name="_")],
                    tail=LogicVar(name="T")
                )
            ]
        ),
        body=[
            LogicStruct(
                functor="member",
                args=[LogicVar(name="X"), LogicVar(name="T")]
            )
        ]
    )
    
    rules = [rule1, rule2]
    
    # Query: member(2, [1,2,3])?
    query_struct = LogicStruct(
        functor="member",
        args=[
            LogicNumber(value=2),
            LogicList(elements=[
                LogicNumber(value=1),
                LogicNumber(value=2),
                LogicNumber(value=3),
            ])
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], rules))
    
    assert len(solutions) >= 1  # Should find at least one solution
    print("✓ List member check works")


def test_list_member_variable_query():
    """Test member/2 with variable to enumerate all members."""
    # Same rules as above
    rule1 = LogicRule(
        head=LogicStruct(
            functor="member",
            args=[
                LogicVar(name="X"),
                LogicList(
                    elements=[LogicVar(name="X")],
                    tail=LogicVar(name="_")
                )
            ]
        ),
        body=[]
    )
    
    rule2 = LogicRule(
        head=LogicStruct(
            functor="member",
            args=[
                LogicVar(name="X"),
                LogicList(
                    elements=[LogicVar(name="_")],
                    tail=LogicVar(name="T")
                )
            ]
        ),
        body=[
            LogicStruct(
                functor="member",
                args=[LogicVar(name="X"), LogicVar(name="T")]
            )
        ]
    )
    
    rules = [rule1, rule2]
    
    # Query: member(X, [a,b,c])? - enumerate all members
    query_struct = LogicStruct(
        functor="member",
        args=[
            LogicVar(name="X"),
            LogicList(elements=[
                LogicAtom(value="a"),
                LogicAtom(value="b"),
                LogicAtom(value="c"),
            ])
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], rules))
    
    assert len(solutions) == 3
    values = [sol.apply(LogicVar(name="X")).value for sol in solutions]
    assert "a" in values
    assert "b" in values
    assert "c" in values
    
    print("✓ List member enumeration works")


def test_list_append():
    """Test append/3 predicate."""
    # Rule 1: append([], L, L).
    rule1 = LogicRule(
        head=LogicStruct(
            functor="append",
            args=[
                LogicList(elements=[]),
                LogicVar(name="L"),
                LogicVar(name="L")
            ]
        ),
        body=[]
    )
    
    # Rule 2: append([H|T], L2, [H|R]) :- append(T, L2, R).
    rule2 = LogicRule(
        head=LogicStruct(
            functor="append",
            args=[
                LogicList(
                    elements=[LogicVar(name="H")],
                    tail=LogicVar(name="T")
                ),
                LogicVar(name="L2"),
                LogicList(
                    elements=[LogicVar(name="H")],
                    tail=LogicVar(name="R")
                )
            ]
        ),
        body=[
            LogicStruct(
                functor="append",
                args=[
                    LogicVar(name="T"),
                    LogicVar(name="L2"),
                    LogicVar(name="R")
                ]
            )
        ]
    )
    
    rules = [rule1, rule2]
    
    # Query: append([1,2], [3,4], X)?
    query_struct = LogicStruct(
        functor="append",
        args=[
            LogicList(elements=[
                LogicNumber(value=1),
                LogicNumber(value=2),
            ]),
            LogicList(elements=[
                LogicNumber(value=3),
                LogicNumber(value=4),
            ]),
            LogicVar(name="X")
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], rules))
    
    assert len(solutions) >= 1
    x_value = solutions[0].apply(LogicVar(name="X"))
    assert isinstance(x_value, LogicList)
    
    # Result is [1|[2|[3,4]]] - a nested tail structure
    # This is correct Prolog behavior when building lists with [H|R] patterns
    # We need to flatten it to verify the values
    def flatten_list(lst: LogicList) -> list:
        """Flatten a LogicList with tail structure into a Python list."""
        result = []
        current = lst
        while isinstance(current, LogicList):
            result.extend(current.elements)
            current = current.tail
        return result
    
    flat_values = flatten_list(x_value)
    assert len(flat_values) == 4
    assert flat_values[0].value == 1
    assert flat_values[1].value == 2
    assert flat_values[2].value == 3
    assert flat_values[3].value == 4
    
    print("✓ List append works")


def test_list_length():
    """Test length/2 predicate."""
    # Rule 1: length([], 0).
    rule1 = LogicRule(
        head=LogicStruct(
            functor="length",
            args=[
                LogicList(elements=[]),
                LogicNumber(value=0)
            ]
        ),
        body=[]
    )
    
    # Rule 2: length([_|T], N) :- length(T, N1), N is N1 + 1.
    # For simplicity, we'll just test with a fixed-length list
    # and not implement arithmetic
    
    # Query: length([a,b,c], X)?
    # We'll test a simpler version: just check empty list
    query_struct = LogicStruct(
        functor="length",
        args=[
            LogicList(elements=[]),
            LogicVar(name="X")
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [], [rule1]))
    
    assert len(solutions) == 1
    x_value = solutions[0].apply(LogicVar(name="X"))
    assert isinstance(x_value, LogicNumber)
    assert x_value.value == 0
    
    print("✓ List length (base case) works")


def test_list_pattern_in_fact():
    """Test facts with list patterns."""
    # Fact: first_two([1,2|_]).
    fact = LogicFact(
        head=LogicStruct(
            functor="first_two",
            args=[
                LogicList(
                    elements=[
                        LogicNumber(value=1),
                        LogicNumber(value=2)
                    ],
                    tail=LogicVar(name="_")
                )
            ]
        )
    )
    
    # Query: first_two([1,2,3,4])?
    query_struct = LogicStruct(
        functor="first_two",
        args=[
            LogicList(elements=[
                LogicNumber(value=1),
                LogicNumber(value=2),
                LogicNumber(value=3),
                LogicNumber(value=4),
            ])
        ]
    )
    
    engine = LogicEngine()
    solutions = list(engine.solve([query_struct], [fact], []))
    
    assert len(solutions) == 1
    print("✓ List patterns in facts work")


# ============================================================================
# Resource Limit Tests
# ============================================================================

def test_list_pattern_respects_depth_limit():
    """Test that list pattern matching respects recursion depth limit."""
    # Recursive rule that processes lists
    rule = LogicRule(
        head=LogicStruct(
            functor="process",
            args=[
                LogicList(
                    elements=[LogicVar(name="H")],
                    tail=LogicVar(name="T")
                )
            ]
        ),
        body=[
            LogicStruct(
                functor="process",
                args=[LogicVar(name="T")]
            )
        ]
    )
    
    # Create a very long list
    long_list = LogicList(elements=[LogicNumber(value=i) for i in range(200)])
    
    query_struct = LogicStruct(
        functor="process",
        args=[long_list]
    )
    
    # Use a low depth limit
    config = LogicEngineConfig(max_depth=50)
    engine = LogicEngine(config=config)
    
    # Should hit depth limit
    from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngineDepthLimit
    
    with pytest.raises(LogicEngineDepthLimit):
        list(engine.solve([query_struct], [], [rule]))
    
    print("✓ List pattern matching respects depth limit")


def test_list_pattern_respects_step_limit():
    """Test that list pattern matching respects step limit."""
    # Rule that generates infinite solutions
    rule = LogicRule(
        head=LogicStruct(
            functor="gen",
            args=[
                LogicList(
                    elements=[LogicVar(name="H")],
                    tail=LogicVar(name="T")
                )
            ]
        ),
        body=[
            LogicStruct(
                functor="gen",
                args=[LogicVar(name="T")]
            )
        ]
    )
    
    query_struct = LogicStruct(
        functor="gen",
        args=[LogicVar(name="X")]
    )
    
    # Use a low step limit
    config = LogicEngineConfig(max_steps=100)
    engine = LogicEngine(config=config)
    
    # Should hit step limit
    from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngineStepLimit
    
    with pytest.raises(LogicEngineStepLimit):
        list(engine.solve([query_struct], [], [rule]))
    
    print("✓ List pattern matching respects step limit")


# ============================================================================
# Compiler Integration Tests
# ============================================================================

def test_query_compiler_with_list_patterns():
    """Test that query compiler properly handles list patterns."""
    # Create knowledge module
    rule = LogicRule(
        head=LogicStruct(
            functor="first",
            args=[
                LogicList(
                    elements=[LogicVar(name="X")],
                    tail=LogicVar(name="_")
                ),
                LogicVar(name="X")
            ]
        ),
        body=[]
    )
    
    module = create_test_module("list_ops", facts=[], rules=[rule])
    
    # Create query
    query = LogicQuery(
        name="test_query",
        goals=[
            LogicStruct(
                functor="first",
                args=[
                    LogicList(elements=[
                        LogicAtom(value="a"),
                        LogicAtom(value="b"),
                    ]),
                    LogicVar(name="Result")
                ]
            )
        ],
        knowledge_sources=["list_ops"],
        variables=["Result"]
    )
    
    # Compile and execute
    context = QueryContext(knowledge_modules={"list_ops": module})
    compiler = QueryCompiler(context)
    compiled = compiler.compile_query(query)
    
    results = compiled.execute_all()
    
    assert len(results) == 1
    assert results[0]["Result"] == "a"
    
    print("✓ Query compiler integration works")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    test_list_head_extraction()
    test_list_tail_extraction()
    test_list_member_check()
    test_list_member_variable_query()
    test_list_append()
    test_list_length()
    test_list_pattern_in_fact()
    
    test_list_pattern_respects_depth_limit()
    test_list_pattern_respects_step_limit()
    
    test_query_compiler_with_list_patterns()
    
    print("\n" + "="*60)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*60)
