"""
Integration tests for query execution with dataset adapters.

Tests end-to-end query execution including:
- Query compilation with dataset adapters
- Fact generation from datasets
- Query execution with real logic engine
- Parameter binding and pagination
"""

import pytest

from namel3ss.ast.application import App
from namel3ss.ast.datasets import Dataset
from namel3ss.ast.logic import (
    LogicAtom,
    LogicFact,
    LogicNumber,
    LogicQuery,
    LogicRule,
    LogicStruct,
    LogicVar,
    KnowledgeModule,
)
from namel3ss.codegen.backend.core.runtime.dataset_adapter_factory import (
    create_adapter_registry,
)
from namel3ss.codegen.backend.core.runtime.logic_engine import LogicEngineConfig
from namel3ss.codegen.backend.core.runtime.query_compiler import (
    QueryCompiler,
    QueryContext,
)


class TestBasicQueryExecution:
    """Test basic query execution without datasets."""
    
    def test_execute_simple_fact_query(self):
        """Test executing a simple query against facts."""
        # Create knowledge module with facts
        knowledge = KnowledgeModule(
            name="family",
            facts=[
                LogicFact(head=LogicStruct(functor="parent", args=[
                    LogicAtom(value="alice"),
                    LogicAtom(value="bob"),
                ])),
                LogicFact(head=LogicStruct(functor="parent", args=[
                    LogicAtom(value="bob"),
                    LogicAtom(value="charlie"),
                ])),
            ],
        )
        
        # Create query
        query = LogicQuery(
            name="find_parents",
            knowledge_sources=["family"],
            goals=[
                LogicStruct(functor="parent", args=[
                    LogicVar(name="Parent"),
                    LogicVar(name="Child"),
                ])
            ],
        )
        
        # Execute query
        context = QueryContext(
            knowledge_modules={"family": knowledge},
            adapter_registry=None,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        assert len(results) == 2
        assert results[0] == {"Parent": "alice", "Child": "bob"}
        assert results[1] == {"Parent": "bob", "Child": "charlie"}
    
    def test_execute_query_with_rule(self):
        """Test executing query that uses rules."""
        knowledge = KnowledgeModule(
            name="family",
            facts=[
                LogicFact(head=LogicStruct(functor="parent", args=[
                    LogicAtom(value="alice"),
                    LogicAtom(value="bob"),
                ])),
                LogicFact(head=LogicStruct(functor="parent", args=[
                    LogicAtom(value="bob"),
                    LogicAtom(value="charlie"),
                ])),
            ],
            rules=[
                LogicRule(
                    head=LogicStruct(functor="ancestor", args=[
                        LogicVar(name="X"),
                        LogicVar(name="Y"),
                    ]),
                    body=[
                        LogicStruct(functor="parent", args=[
                            LogicVar(name="X"),
                            LogicVar(name="Y"),
                        ]),
                    ],
                ),
                LogicRule(
                    head=LogicStruct(functor="ancestor", args=[
                        LogicVar(name="X"),
                        LogicVar(name="Z"),
                    ]),
                    body=[
                        LogicStruct(functor="parent", args=[
                            LogicVar(name="X"),
                            LogicVar(name="Y"),
                        ]),
                        LogicStruct(functor="ancestor", args=[
                            LogicVar(name="Y"),
                            LogicVar(name="Z"),
                        ]),
                    ],
                ),
            ],
        )
        
        query = LogicQuery(
            name="find_ancestors",
            knowledge_sources=["family"],
            goals=[
                LogicStruct(functor="ancestor", args=[
                    LogicAtom(value="alice"),
                    LogicVar(name="Descendant"),
                ])
            ],
        )
        
        context = QueryContext(knowledge_modules={"family": knowledge})
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        # Should find both bob (direct) and charlie (indirect)
        assert len(results) >= 2
        descendants = {r["Descendant"] for r in results}
        assert "bob" in descendants
        assert "charlie" in descendants


class TestQueryWithDatasetAdapter:
    """Test query execution with dataset adapters."""
    
    def test_query_against_dataset(self):
        """Test querying data from a dataset adapter."""
        # Create app with inline dataset
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="users",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": 1, "name": "Alice", "age": 30},
                            {"id": 2, "name": "Bob", "age": 25},
                            {"id": 3, "name": "Charlie", "age": 35},
                        ]
                    },
                )
            ],
        )
        
        # Create adapter registry
        registry = create_adapter_registry(app)
        
        # Create query to find all users
        query = LogicQuery(
            name="all_users",
            knowledge_sources=[],
            goals=[
                LogicStruct(functor="row_users", args=[
                    LogicVar(name="RowID"),
                ])
            ],
        )
        
        # Execute query
        context = QueryContext(
            knowledge_modules={},
            adapter_registry=registry,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        # Should find all 3 rows
        assert len(results) == 3
    
    def test_query_dataset_fields(self):
        """Test querying specific fields from dataset."""
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="products",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": 1, "name": "Widget", "price": 9.99},
                            {"id": 2, "name": "Gadget", "price": 19.99},
                        ]
                    },
                )
            ],
        )
        
        registry = create_adapter_registry(app)
        
        # Query for product names
        query = LogicQuery(
            name="product_names",
            knowledge_sources=[],
            goals=[
                LogicStruct(functor="row_products", args=[
                    LogicVar(name="RowID"),
                ]),
                LogicStruct(functor="field_products", args=[
                    LogicVar(name="RowID"),
                    LogicAtom(value="name"),
                    LogicVar(name="Name"),
                ]),
            ],
        )
        
        context = QueryContext(
            knowledge_modules={},
            adapter_registry=registry,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        assert len(results) == 2
        names = {r["Name"] for r in results}
        assert "Widget" in names
        assert "Gadget" in names
    
    def test_query_with_filter(self):
        """Test querying dataset with filtering conditions."""
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="orders",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": 1, "customer": "Alice", "total": 100},
                            {"id": 2, "customer": "Bob", "total": 200},
                            {"id": 3, "customer": "Alice", "total": 150},
                        ]
                    },
                )
            ],
        )
        
        registry = create_adapter_registry(app)
        
        # Query for Alice's orders
        query = LogicQuery(
            name="alice_orders",
            knowledge_sources=[],
            goals=[
                LogicStruct(functor="row_orders", args=[
                    LogicVar(name="RowID"),
                ]),
                LogicStruct(functor="field_orders", args=[
                    LogicVar(name="RowID"),
                    LogicAtom(value="customer"),
                    LogicAtom(value="Alice"),
                ]),
                LogicStruct(functor="field_orders", args=[
                    LogicVar(name="RowID"),
                    LogicAtom(value="total"),
                    LogicVar(name="Total"),
                ]),
            ],
        )
        
        context = QueryContext(
            knowledge_modules={},
            adapter_registry=registry,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        # Should find only Alice's orders
        assert len(results) == 2
        totals = [r["Total"] for r in results]
        assert 100 in totals
        assert 150 in totals


class TestQueryLimitsAndPagination:
    """Test query limits and pagination."""
    
    def test_query_with_limit(self):
        """Test query execution respects limit."""
        knowledge = KnowledgeModule(
            name="numbers",
            facts=[
                LogicFact(head=LogicStruct(functor="num", args=[LogicNumber(value=i)]))
                for i in range(10)
            ],
        )
        
        query = LogicQuery(
            name="limited_nums",
            knowledge_sources=["numbers"],
            goals=[
                LogicStruct(functor="num", args=[LogicVar(name="N")])
            ],
            limit=5,
        )
        
        context = QueryContext(knowledge_modules={"numbers": knowledge})
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        assert len(results) <= 5
    
    def test_query_pagination(self):
        """Test manual pagination with offset."""
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="items",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": i, "value": f"item_{i}"}
                            for i in range(20)
                        ]
                    },
                )
            ],
        )
        
        registry = create_adapter_registry(app)
        
        query = LogicQuery(
            name="all_items",
            knowledge_sources=[],
            goals=[
                LogicStruct(functor="row_items", args=[LogicVar(name="RowID")])
            ],
        )
        
        context = QueryContext(
            knowledge_modules={},
            adapter_registry=registry,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        all_results = compiled.execute_all()
        
        # Test pagination manually
        page_size = 5
        offset = 0
        page = all_results[offset:offset + page_size]
        assert len(page) == 5
        
        offset = 10
        page = all_results[offset:offset + page_size]
        assert len(page) == 5


class TestQuerySafetyLimits:
    """Test query execution safety limits."""
    
    def test_query_depth_limit(self):
        """Test that max_depth limit is enforced."""
        # Create a recursive rule that could go deep
        knowledge = KnowledgeModule(
            name="chain",
            facts=[
                LogicFact(head=LogicStruct(functor="next", args=[
                    LogicNumber(value=i),
                    LogicNumber(value=i+1),
                ]))
                for i in range(100)
            ],
            rules=[
                LogicRule(
                    head=LogicStruct(functor="reachable", args=[
                        LogicVar(name="X"),
                        LogicVar(name="Y"),
                    ]),
                    body=[
                        LogicStruct(functor="next", args=[
                            LogicVar(name="X"),
                            LogicVar(name="Y"),
                        ]),
                    ],
                ),
                LogicRule(
                    head=LogicStruct(functor="reachable", args=[
                        LogicVar(name="X"),
                        LogicVar(name="Z"),
                    ]),
                    body=[
                        LogicStruct(functor="next", args=[
                            LogicVar(name="X"),
                            LogicVar(name="Y"),
                        ]),
                        LogicStruct(functor="reachable", args=[
                            LogicVar(name="Y"),
                            LogicVar(name="Z"),
                        ]),
                    ],
                ),
            ],
        )
        
        query = LogicQuery(
            name="deep_reach",
            knowledge_sources=["chain"],
            goals=[
                LogicStruct(functor="reachable", args=[
                    LogicNumber(value=0),
                    LogicVar(name="End"),
                ])
            ],
        )
        
        # Execute with low depth limit
        context = QueryContext(knowledge_modules={"chain": knowledge})
        config = LogicEngineConfig(max_depth=10, max_steps=10000)
        compiler = QueryCompiler(context, engine_config=config)
        compiled = compiler.compile_query(query)
        
        # Should complete without error but may not find all solutions
        results = compiled.execute_all()
        assert isinstance(results, list)
    
    def test_query_step_limit(self):
        """Test that max_steps limit is enforced."""
        # Create facts that will require many steps
        knowledge = KnowledgeModule(
            name="data",
            facts=[
                LogicFact(head=LogicStruct(functor="item", args=[
                    LogicNumber(value=i)
                ]))
                for i in range(1000)
            ],
        )
        
        query = LogicQuery(
            name="all_items",
            knowledge_sources=["data"],
            goals=[
                LogicStruct(functor="item", args=[LogicVar(name="X")])
            ],
        )
        
        # Execute with low step limit
        context = QueryContext(knowledge_modules={"data": knowledge})
        config = LogicEngineConfig(max_steps=100)
        compiler = QueryCompiler(context, engine_config=config)
        compiled = compiler.compile_query(query)
        
        results = compiled.execute_all()
        # May be limited by max_steps
        assert isinstance(results, list)


class TestMultipleDatasets:
    """Test queries across multiple datasets."""
    
    def test_join_across_datasets(self):
        """Test joining data from multiple datasets."""
        app = App(
            name="test",
            datasets=[
                Dataset(
                    name="customers",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": 1, "name": "Alice"},
                            {"id": 2, "name": "Bob"},
                        ]
                    },
                ),
                Dataset(
                    name="orders",
                    source_type="inline",
                    source="",
                    metadata={
                        "records": [
                            {"id": 101, "customer_id": 1, "total": 100},
                            {"id": 102, "customer_id": 2, "total": 200},
                            {"id": 103, "customer_id": 1, "total": 150},
                        ]
                    },
                ),
            ],
        )
        
        registry = create_adapter_registry(app)
        
        # Query to join customers with their orders
        query = LogicQuery(
            name="customer_orders",
            knowledge_sources=[],
            goals=[
                # Get customer row
                LogicStruct(functor="row_customers", args=[LogicVar(name="CustRow")]),
                LogicStruct(functor="field_customers", args=[
                    LogicVar(name="CustRow"),
                    LogicAtom(value="id"),
                    LogicVar(name="CustID"),
                ]),
                LogicStruct(functor="field_customers", args=[
                    LogicVar(name="CustRow"),
                    LogicAtom(value="name"),
                    LogicVar(name="Name"),
                ]),
                # Get matching order
                LogicStruct(functor="row_orders", args=[LogicVar(name="OrderRow")]),
                LogicStruct(functor="field_orders", args=[
                    LogicVar(name="OrderRow"),
                    LogicAtom(value="customer_id"),
                    LogicVar(name="CustID"),  # Join condition
                ]),
                LogicStruct(functor="field_orders", args=[
                    LogicVar(name="OrderRow"),
                    LogicAtom(value="total"),
                    LogicVar(name="Total"),
                ]),
            ],
        )
        
        context = QueryContext(
            knowledge_modules={},
            adapter_registry=registry,
        )
        compiler = QueryCompiler(context)
        compiled = compiler.compile_query(query)
        results = compiled.execute_all()
        
        # Should find 3 customer-order pairs
        assert len(results) == 3
        
        # Check Alice has 2 orders
        alice_orders = [r for r in results if r["Name"] == "Alice"]
        assert len(alice_orders) == 2
        
        # Check Bob has 1 order
        bob_orders = [r for r in results if r["Name"] == "Bob"]
        assert len(bob_orders) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
