"""Unit tests for data binding AST nodes."""

import pytest
from namel3ss.ast.pages import ShowTable, ShowChart, ShowForm, DataBindingConfig
from namel3ss.ast.datasets import Dataset, DatasetSchemaField, DatasetAccessPolicy


class TestDataBindingConfig:
    """Test DataBindingConfig AST node."""
    
    def test_default_values(self):
        """DataBindingConfig has sensible defaults."""
        config = DataBindingConfig()
        
        assert config.auto_refresh is False
        assert config.refresh_interval is None
        assert config.page_size == 50
        assert config.enable_sorting is True
        assert config.enable_filtering is True
        assert config.enable_search is False
        assert config.cache_ttl is None
        assert config.editable is False
        assert config.enable_create is False
        assert config.enable_update is False
        assert config.enable_delete is False
        assert config.subscribe_to_changes is False
        assert config.field_mapping == {}
        assert config.write_endpoint is None
        assert config.optimistic_updates is True
    
    def test_custom_values(self):
        """DataBindingConfig accepts custom values."""
        config = DataBindingConfig(
            auto_refresh=True,
            refresh_interval=30,
            page_size=20,
            enable_sorting=False,
            enable_filtering=False,
            enable_search=True,
            cache_ttl=600,
            editable=True,
            enable_create=True,
            enable_update=True,
            enable_delete=True,
            subscribe_to_changes=True,
            field_mapping={"user_name": "name"},
            write_endpoint="/custom/endpoint",
            optimistic_updates=False,
        )
        
        assert config.auto_refresh is True
        assert config.refresh_interval == 30
        assert config.page_size == 20
        assert config.enable_sorting is False
        assert config.enable_filtering is False
        assert config.enable_search is True
        assert config.cache_ttl == 600
        assert config.editable is True
        assert config.enable_create is True
        assert config.enable_update is True
        assert config.enable_delete is True
        assert config.subscribe_to_changes is True
        assert config.field_mapping == {"user_name": "name"}
        assert config.write_endpoint == "/custom/endpoint"
        assert config.optimistic_updates is False
    
    def test_read_only_config(self):
        """DataBindingConfig can represent read-only binding."""
        config = DataBindingConfig(
            editable=False,
            enable_create=False,
            enable_update=False,
            enable_delete=False,
        )
        
        assert not config.editable
        assert not config.enable_create
        assert not config.enable_update
        assert not config.enable_delete
    
    def test_full_crud_config(self):
        """DataBindingConfig can represent full CRUD binding."""
        config = DataBindingConfig(
            editable=True,
            enable_create=True,
            enable_update=True,
            enable_delete=True,
        )
        
        assert config.editable
        assert config.enable_create
        assert config.enable_update
        assert config.enable_delete


class TestDatasetAccessPolicy:
    """Test DatasetAccessPolicy AST node."""
    
    def test_default_values(self):
        """DatasetAccessPolicy has conservative defaults."""
        policy = DatasetAccessPolicy()
        
        # Default is read-only for safety
        assert policy.read_only is True
        assert policy.allow_create is False
        assert policy.allow_update is False
        assert policy.allow_delete is False
        assert policy.primary_key is None
        assert policy.required_capabilities == []
    
    def test_read_only_policy(self):
        """DatasetAccessPolicy can enforce read-only access."""
        policy = DatasetAccessPolicy(read_only=True)
        
        assert policy.read_only is True
        # Note: read_only should be checked first in enforcement
    
    def test_restricted_policy(self):
        """DatasetAccessPolicy can restrict specific operations."""
        policy = DatasetAccessPolicy(
            read_only=False,
            allow_create=True,
            allow_update=True,
            allow_delete=False,
        )
        
        assert not policy.read_only
        assert policy.allow_create
        assert policy.allow_update
        assert not policy.allow_delete
    
    def test_capability_requirements(self):
        """DatasetAccessPolicy can require capabilities."""
        policy = DatasetAccessPolicy(
            required_capabilities=["user.manage", "data.write"]
        )
        
        assert policy.required_capabilities == ["user.manage", "data.write"]
    
    def test_primary_key_specification(self):
        """DatasetAccessPolicy can specify primary key."""
        policy = DatasetAccessPolicy(primary_key="id")
        
        assert policy.primary_key == "id"
    
    def test_complete_policy(self):
        """DatasetAccessPolicy with all fields."""
        policy = DatasetAccessPolicy(
            read_only=False,
            allow_create=True,
            allow_update=True,
            allow_delete=False,
            primary_key="user_id",
            required_capabilities=["admin.access"],
        )
        
        assert not policy.read_only
        assert policy.allow_create
        assert policy.allow_update
        assert not policy.allow_delete
        assert policy.primary_key == "user_id"
        assert policy.required_capabilities == ["admin.access"]


class TestShowTableWithBinding:
    """Test ShowTable with DataBindingConfig."""
    
    def test_table_without_binding(self):
        """ShowTable works without binding config."""
        table = ShowTable(title="Users", source_type="dataset", source="users_dataset")
        
        assert table.title == "Users"
        assert table.binding is None
    
    def test_table_with_binding(self):
        """ShowTable accepts binding config."""
        binding = DataBindingConfig(
            editable=True,
            page_size=20,
        )
        table = ShowTable(title="Users", source_type="dataset", source="UserData", binding=binding)
        
        assert table.title == "Users"
        assert table.binding is not None
        assert table.binding.editable is True
        assert table.binding.page_size == 20
    
    def test_table_with_inline_binding(self):
        """ShowTable with inline binding config."""
        table = ShowTable(
            title="Products",
            source_type="dataset",
            source="Products",
            binding=DataBindingConfig(
                auto_refresh=True,
                subscribe_to_changes=True,
            )
        )
        
        assert table.source == "Products"
        assert table.binding.auto_refresh is True
        assert table.binding.subscribe_to_changes is True


class TestShowChartWithBinding:
    """Test ShowChart with DataBindingConfig."""
    
    def test_chart_without_binding(self):
        """ShowChart works without binding config."""
        chart = ShowChart(heading="Sales", source_type="dataset", source="sales_data", chart_type="line")
        
        assert chart.heading == "Sales"
        assert chart.binding is None
    
    def test_chart_with_binding(self):
        """ShowChart accepts binding config."""
        binding = DataBindingConfig(
            auto_refresh=True,
            subscribe_to_changes=True,
        )
        chart = ShowChart(heading="Sales", source_type="dataset", source="SalesData", chart_type="line", binding=binding)
        
        assert chart.heading == "Sales"
        assert chart.binding is not None
        assert chart.binding.subscribe_to_changes is True


class TestShowFormWithBinding:
    """Test ShowForm with DataBindingConfig."""
    
    def test_form_without_binding(self):
        """ShowForm works without binding config."""
        form = ShowForm(title="User Form")
        
        assert form.title == "User Form"
        assert form.binding is None
    
    def test_form_with_binding(self):
        """ShowForm accepts binding config."""
        binding = DataBindingConfig(
            editable=True,
            enable_create=True,
            enable_update=True,
        )
        form = ShowForm(title="User Form", binding=binding, bound_dataset="UserData")
        
        assert form.title == "User Form"
        assert form.binding is not None
        assert form.binding.editable is True
        assert form.binding.enable_create is True
        assert form.binding.enable_update is True
        assert form.bound_dataset == "UserData"


class TestDatasetWithAccessPolicy:
    """Test Dataset with DatasetAccessPolicy."""
    
    def test_dataset_without_policy(self):
        """Dataset works without access policy."""
        dataset = Dataset(
            name="Users",
            source_type="sql",
            source="users_table",
            schema=[
                DatasetSchemaField(name="id", dtype="integer"),
                DatasetSchemaField(name="name", dtype="string"),
            ]
        )
        
        assert dataset.name == "Users"
        assert dataset.access_policy is None
    
    def test_dataset_with_policy(self):
        """Dataset accepts access policy."""
        policy = DatasetAccessPolicy(
            read_only=False,
            allow_create=True,
            allow_update=True,
            allow_delete=False,
            primary_key="id",
        )
        dataset = Dataset(
            name="Users",
            source_type="sql",
            source="users_table",
            schema=[
                DatasetSchemaField(name="id", dtype="integer"),
                DatasetSchemaField(name="name", dtype="string"),
            ],
            access_policy=policy,
        )
        
        assert dataset.name == "Users"
        assert dataset.access_policy is not None
        assert dataset.access_policy.primary_key == "id"
        assert not dataset.access_policy.allow_delete
    
    def test_dataset_with_inline_policy(self):
        """Dataset with inline access policy."""
        dataset = Dataset(
            name="Products",
            source_type="sql",
            source="products_table",
            schema=[
                DatasetSchemaField(name="id", dtype="integer"),
                DatasetSchemaField(name="title", dtype="string"),
                DatasetSchemaField(name="price", dtype="float"),
            ],
            access_policy=DatasetAccessPolicy(
                read_only=True,
                required_capabilities=["products.read"],
            )
        )
        
        assert dataset.access_policy.read_only is True
        assert dataset.access_policy.required_capabilities == ["products.read"]


class TestBindingConfigEdgeCases:
    """Test edge cases and validation scenarios."""
    
    def test_editable_without_update_permission(self):
        """Editable flag without enable_update is valid (UI-only)."""
        config = DataBindingConfig(
            editable=True,
            enable_update=False,
        )
        
        # This is valid - editable might be used for UI dirty state
        assert config.editable is True
        assert config.enable_update is False
    
    def test_large_page_size(self):
        """DataBindingConfig accepts large page sizes."""
        config = DataBindingConfig(page_size=1000)
        
        assert config.page_size == 1000
    
    def test_complex_field_mapping(self):
        """DataBindingConfig with complex field mapping."""
        config = DataBindingConfig(
            field_mapping={
                "user_name": "name",
                "user_email": "email",
                "created_at": "timestamp",
            }
        )
        
        assert len(config.field_mapping) == 3
        assert config.field_mapping["user_name"] == "name"
        assert config.field_mapping["user_email"] == "email"
        assert config.field_mapping["created_at"] == "timestamp"
    
    def test_multiple_capabilities(self):
        """DatasetAccessPolicy with multiple capabilities."""
        policy = DatasetAccessPolicy(
            required_capabilities=[
                "data.read",
                "data.write",
                "admin.access",
                "special.permission",
            ]
        )
        
        assert len(policy.required_capabilities) == 4
        assert "admin.access" in policy.required_capabilities
