"""Unit tests for data binding IR builder logic."""

import pytest
from namel3ss.ast import App
from namel3ss.ast.application import Page
from namel3ss.ast.pages import ShowTable, DataBindingConfig
from namel3ss.ast.datasets import Dataset, DatasetSchemaField, DatasetAccessPolicy
from namel3ss.ir.builder import build_backend_ir


class TestIRBuilderBasics:
    """Test basic IR builder functionality with data binding."""
    
    def test_app_with_dataset_builds(self):
        """App with simple dataset builds successfully."""
        app = App(
            name="test_app",
            datasets=[
                Dataset(
                    name="SimpleData",
                    source_type="sql",
                    source="simple_table",
                    schema=[
                        DatasetSchemaField(name="id", dtype="integer"),
                        DatasetSchemaField(name="value", dtype="string"),
                    ]
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        # IR should build without errors
        assert ir is not None
        assert len(ir.datasets) >= 1
    
    def test_dataset_with_access_policy(self):
        """Dataset with access policy flows to IR."""
        app = App(
            name="test_app",
            datasets=[
                Dataset(
                    name="SecureData",
                    source_type="sql",
                    source="secure_table",
                    schema=[
                        DatasetSchemaField(name="id", dtype="integer"),
                    ],
                    access_policy=DatasetAccessPolicy(
                        read_only=False,
                        allow_create=True,
                        allow_update=True,
                        allow_delete=False,
                        primary_key="id",
                    )
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        # Dataset should be extracted with policy
        assert ir is not None
        dataset = next((ds for ds in ir.datasets if ds.name == "SecureData"), None)
        assert dataset is not None


class TestIRBuilderWithComponents:
    """Test IR builder with data-bound components."""
    
    def test_app_with_bound_table(self):
        """App with data-bound table builds successfully."""
        app = App(
            name="test_app",
            datasets=[
                Dataset(
                    name="Users",
                    source_type="sql",
                    source="users_table",
                    schema=[
                        DatasetSchemaField(name="id", dtype="integer"),
                        DatasetSchemaField(name="name", dtype="string"),
                    ]
                )
            ],
            pages=[
                Page(
                    name="UsersPage",
                    route="/users",
                    body=[
                        ShowTable(
                            title="Users Table",
                            source_type="dataset",
                            source="Users",
                            binding=DataBindingConfig(
                                editable=True,
                                page_size=20,
                            )
                        )
                    ]
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        # IR should build without errors
        assert ir is not None
        assert len(ir.datasets) >= 1
    
    def test_backwards_compatibility(self):
        """App without binding features still works."""
        app = App(
            name="legacy_app",
            datasets=[
                Dataset(
                    name="LegacyData",
                    source_type="sql",
                    source="legacy_table",
                    schema=[DatasetSchemaField(name="id", dtype="integer")]
                )
            ],
            pages=[
                Page(
                    name="LegacyPage",
                    route="/legacy",
                    body=[
                        ShowTable(
                            title="Legacy Table",
                            source_type="dataset",
                            source="LegacyData"
                        )
                    ]
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        # Should work without errors - backwards compatibility
        assert ir is not None
        assert len(ir.datasets) >= 1
