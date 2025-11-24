"""Unit tests for data binding IR specifications."""

import pytest
from namel3ss.ir.spec import DataBindingSpec, UpdateChannelSpec


class TestDataBindingSpec:
    """Test DataBindingSpec IR node."""
    
    def test_minimal_binding_spec(self):
        """DataBindingSpec with minimal required fields."""
        spec = DataBindingSpec(
            dataset_name="UserData",
            endpoint_path="/api/datasets/UserData"
        )
        
        assert spec.dataset_name == "UserData"
        assert spec.endpoint_path == "/api/datasets/UserData"
        assert spec.page_size == 50
        assert spec.enable_sorting is True
        assert spec.enable_filtering is True
        assert spec.enable_search is False
        assert spec.editable is False
        assert spec.enable_create is False
        assert spec.enable_update is False
        assert spec.enable_delete is False
        assert spec.subscribe_to_changes is False
        assert spec.optimistic_updates is True
        assert spec.field_mapping == {}
    
    def test_complete_binding_spec(self):
        """DataBindingSpec with all fields."""
        spec = DataBindingSpec(
            dataset_name="Products",
            endpoint_path="/api/datasets/Products",
            page_size=25,
            enable_sorting=True,
            sortable_fields=["name", "price"],
            enable_filtering=True,
            filterable_fields=["category"],
            enable_search=True,
            searchable_fields=["name", "description"],
            editable=True,
            enable_create=True,
            enable_update=True,
            enable_delete=True,
            create_endpoint="/api/datasets/Products",
            update_endpoint="/api/datasets/Products/{id}",
            delete_endpoint="/api/datasets/Products/{id}",
            subscribe_to_changes=True,
            websocket_topic="dataset:Products:changes",
            polling_interval=30,
            cache_ttl=60,
            optimistic_updates=False,
            field_mapping={"product_name": "name"},
        )
        
        assert spec.dataset_name == "Products"
        assert spec.page_size == 25
        assert spec.sortable_fields == ["name", "price"]
        assert spec.searchable_fields == ["name", "description"]
        assert spec.editable is True
        assert spec.enable_create is True
        assert spec.enable_update is True
        assert spec.enable_delete is True
        assert spec.create_endpoint == "/api/datasets/Products"
        assert spec.update_endpoint == "/api/datasets/Products/{id}"
        assert spec.delete_endpoint == "/api/datasets/Products/{id}"
        assert spec.subscribe_to_changes is True
        assert spec.websocket_topic == "dataset:Products:changes"
        assert spec.polling_interval == 30
        assert spec.cache_ttl == 60
        assert spec.optimistic_updates is False
        assert spec.field_mapping == {"product_name": "name"}
    
    def test_read_only_binding_spec(self):
        """DataBindingSpec for read-only dataset."""
        spec = DataBindingSpec(
            dataset_name="Reports",
            endpoint_path="/api/datasets/Reports",
            editable=False,
            enable_create=False,
            enable_update=False,
            enable_delete=False,
        )
        
        assert spec.dataset_name == "Reports"
        assert not spec.editable
        assert not spec.enable_create
        assert not spec.enable_update
        assert not spec.enable_delete
        assert spec.create_endpoint is None
        assert spec.update_endpoint is None
        assert spec.delete_endpoint is None


class TestUpdateChannelSpec:
    """Test UpdateChannelSpec IR node."""
    
    def test_minimal_channel_spec(self):
        """UpdateChannelSpec with minimal required fields."""
        spec = UpdateChannelSpec(
            name="main_channel",
            dataset_name="UserData",
        )
        
        assert spec.name == "main_channel"
        assert spec.dataset_name == "UserData"
        assert spec.event_types == ["create", "update", "delete"]
        assert spec.transport == "websocket"
        assert spec.requires_auth is True
        assert spec.required_capabilities == []
        assert spec.redis_channel is None
    
    def test_complete_channel_spec(self):
        """UpdateChannelSpec with all fields."""
        spec = UpdateChannelSpec(
            name="secure_channel",
            dataset_name="SensitiveData",
            event_types=["create", "update"],
            transport="websocket",
            requires_auth=True,
            required_capabilities=["data.read", "data.subscribe"],
            redis_channel="redis:SensitiveData:updates",
        )
        
        assert spec.name == "secure_channel"
        assert spec.dataset_name == "SensitiveData"
        assert spec.event_types == ["create", "update"]
        assert "delete" not in spec.event_types
        assert spec.transport == "websocket"
        assert spec.requires_auth is True
        assert spec.required_capabilities == ["data.read", "data.subscribe"]
        assert spec.redis_channel == "redis:SensitiveData:updates"


class TestBindingSpecDefaults:
    """Test default values."""
    
    def test_boolean_flags_default_safe(self):
        """Boolean flags default to safe/conservative values."""
        spec = DataBindingSpec(
            dataset_name="Safe",
            endpoint_path="/api/datasets/Safe"
        )
        
        # Safe defaults: read-only behavior
        assert spec.editable is False
        assert spec.enable_create is False
        assert spec.enable_update is False
        assert spec.enable_delete is False
        assert spec.subscribe_to_changes is False
        
        # Opt-in features enabled by default
        assert spec.enable_sorting is True
        assert spec.enable_filtering is True
        assert spec.optimistic_updates is True
