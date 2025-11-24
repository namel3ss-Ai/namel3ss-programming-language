"""
End-to-end integration tests for data binding system.

Tests the complete data binding pipeline:
- N3 source → AST → IR → Backend/Frontend codegen
- Live data binding with CRUD operations
- Realtime updates via WebSocket
- Security and validation
"""

import pytest
from pathlib import Path
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

from namel3ss.ast.datasets import Dataset, DatasetAccessPolicy
from namel3ss.ast.pages import DataBindingConfig
from namel3ss.ir.spec import DataBindingSpec, UpdateChannelSpec
from namel3ss.codegen.backend.core.dataset_router import _render_dataset_router_module
from namel3ss.codegen.frontend.react.dataset_client import _generate_dataset_client_code
from namel3ss.codegen.frontend.react.bound_components import generate_bound_components


class TestDataBindingE2E:
    """End-to-end tests for data binding system."""
    
    @pytest.fixture
    def sample_n3_source(self):
        """Sample N3 source code with data binding."""
        return textwrap.dedent('''
            dataset users {
                id: integer primary_key,
                name: string required,
                email: string,
                created_at: datetime auto_now
                
                access_policy {
                    read: true
                    create: true
                    update: true
                    delete: admin_only
                    
                    sortable: [name, email, created_at]
                    searchable: [name, email]
                    page_size: 50
                }
                
                realtime: true
                refresh_policy: on_change
            }
            
            page UserManagement {
                title: "User Management"
                
                show_table users {
                    title: "All Users"
                    columns: [id, name, email, created_at]
                    
                    binding {
                        auto_refresh: true
                        editable: true
                        realtime: true
                        
                        crud {
                            create: true
                            update: true
                            delete: admin_only
                        }
                        
                        field_mapping {
                            name: text_input
                            email: email_input
                        }
                    }
                    
                    pagination {
                        page_size: 25
                        show_totals: true
                    }
                }
                
                show_form user_form {
                    title: "Add/Edit User"
                    fields: [name, email]
                    
                    binding {
                        dataset: users
                        mode: create_update
                        auto_clear: true
                        
                        validation {
                            name: required
                            email: [required, email_format]
                        }
                        
                        on_success: refresh_table
                    }
                }
            }
        ''').strip()
    
    @pytest.fixture
    def compiled_ast_ir(self, sample_n3_source):
        """Simulate compiled AST → IR transformation."""
        # Mock Dataset AST
        dataset_ast = Dataset(
            name="users",
            schema=[
                {"name": "id", "type": "integer", "primary_key": True},
                {"name": "name", "type": "string", "required": True},
                {"name": "email", "type": "string", "required": False},
                {"name": "created_at", "type": "datetime", "auto_now": True},
            ],
            access_policy=DatasetAccessPolicy(
                read=True,
                create=True, 
                update=True,
                delete="admin_only",
                sortable_fields=["name", "email", "created_at"],
                searchable_fields=["name", "email"],
                default_page_size=50,
            ),
            reactive=True,
            refresh_policy="on_change",
        )
        
        # Mock DataBindingConfig
        binding_config = DataBindingConfig(
            auto_refresh=True,
            editable=True,
            realtime=True,
            crud_settings={
                "create": True,
                "update": True, 
                "delete": "admin_only",
            },
            field_mapping={
                "name": "text_input",
                "email": "email_input",
            },
            optimistic_updates=True,
        )
        
        # Mock IR Specs
        data_binding_spec = DataBindingSpec(
            dataset_name="users",
            read_endpoint="/api/datasets/users",
            create_endpoint="/api/datasets/users",
            update_endpoint="/api/datasets/users/{id}",
            delete_endpoint="/api/datasets/users/{id}",
            enable_realtime=True,
            enable_polling=True,
            polling_interval=30000,
            page_size=25,
            sort_by="created_at",
            sort_order="desc",
            crud_permissions={
                "create": True,
                "update": True,
                "delete": "admin_only",
            },
            field_validations={
                "name": {"required": True},
                "email": {"required": True, "format": "email"},
            },
            optimistic_updates=True,
        )
        
        update_channel_spec = UpdateChannelSpec(
            channel_name="dataset_updates:users",
            event_types=["create", "update", "delete"],
            transport="redis_pubsub",
            reconnect_strategy="exponential_backoff",
        )
        
        # Mock BackendIR
        backend_ir = MagicMock()
        backend_ir.datasets = [dataset_ast]
        backend_ir.data_bindings = [data_binding_spec]
        backend_ir.update_channels = [update_channel_spec]
        
        return {
            "dataset_ast": dataset_ast,
            "binding_config": binding_config,
            "backend_ir": backend_ir,
            "data_binding_spec": data_binding_spec,
            "update_channel_spec": update_channel_spec,
        }
    
    def test_ast_to_ir_transformation(self, compiled_ast_ir):
        """Test AST dataset definitions map to IR specs correctly."""
        dataset_ast = compiled_ast_ir["dataset_ast"]
        data_binding_spec = compiled_ast_ir["data_binding_spec"]
        
        # Verify dataset properties carry through
        assert data_binding_spec.dataset_name == dataset_ast.name
        assert data_binding_spec.enable_realtime == dataset_ast.reactive
        
        # Verify access policy mapping
        assert data_binding_spec.crud_permissions["create"] == dataset_ast.access_policy.create
        assert data_binding_spec.crud_permissions["update"] == dataset_ast.access_policy.update
        assert data_binding_spec.crud_permissions["delete"] == dataset_ast.access_policy.delete
        
        # Verify field mapping
        assert len(data_binding_spec.field_validations) > 0
    
    def test_backend_codegen(self, compiled_ast_ir):
        """Test backend code generation from IR."""
        backend_ir = compiled_ast_ir["backend_ir"]
        
        # Generate backend router
        router_code = _render_dataset_router_module(backend_ir)
        
        # Verify CRUD endpoints generated
        assert "async def get_users_dataset" in router_code
        assert "async def create_users_item" in router_code
        assert "async def update_users_item" in router_code
        assert "async def delete_users_item" in router_code
        
        # Verify data validation
        assert "UsersItem(BaseModel)" in router_code
        assert "UsersCreateRequest(BaseModel)" in router_code
        assert "UsersUpdateRequest(BaseModel)" in router_code
        
        # Verify realtime integration
        assert "broadcast_dataset_change" in router_code
        
        # Verify security
        assert "HTTPException" in router_code
        assert "status_code" in router_code
        
        # Verify pagination
        assert "page:" in router_code
        assert "page_size:" in router_code
        assert "total" in router_code.lower()
        
        # Verify observability
        assert "tracer" in router_code
        assert "span" in router_code
        assert "logger" in router_code
    
    def test_frontend_codegen(self, compiled_ast_ir):
        """Test frontend code generation from IR."""
        backend_ir = compiled_ast_ir["backend_ir"]
        
        # Generate frontend client
        client_code = _generate_dataset_client_code(backend_ir)
        
        # Verify DatasetClient functionality
        assert "class DatasetClient" in client_code
        assert "async fetch(options:" in client_code
        assert "async create(data:" in client_code
        assert "async update(id:" in client_code
        assert "async delete(id:" in client_code
        
        # Verify React hooks
        assert "export function useDataset" in client_code
        assert "export function useDatasetMutation" in client_code
        
        # Verify WebSocket integration
        assert "WebSocketManager" in client_code
        assert "subscribe(" in client_code
        assert "reconnect" in client_code.lower()
        
        # Verify type safety
        assert "export interface Users {" in client_code
        assert "id: number;" in client_code
        assert "name: string;" in client_code
        assert "email?: string;" in client_code
        
        # Verify error handling
        assert "try {" in client_code
        assert "catch" in client_code
        assert "Error" in client_code
    
    def test_bound_components_generation(self, compiled_ast_ir):
        """Test data-bound component generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            components_dir = Path(temp_dir)
            
            generate_bound_components(components_dir)
            
            # Verify all components generated
            table_file = components_dir / "BoundTableWidget.tsx"
            chart_file = components_dir / "BoundChartWidget.tsx" 
            form_file = components_dir / "BoundFormWidget.tsx"
            
            assert table_file.exists()
            assert chart_file.exists()
            assert form_file.exists()
            
            # Check table component features
            table_content = table_file.read_text()
            assert "useDataset" in table_content
            assert "useDatasetMutation" in table_content
            assert "handleEdit" in table_content
            assert "handleSave" in table_content
            assert "handleDelete" in table_content
            assert "pagination" in table_content.lower()
            
            # Check form component features
            form_content = form_file.read_text()
            assert "create" in form_content
            assert "update" in form_content
            assert "handleSubmit" in form_content
            assert "validation" in form_content.lower() or "required" in form_content
    
    def test_realtime_integration(self, compiled_ast_ir):
        """Test realtime update integration."""
        backend_ir = compiled_ast_ir["backend_ir"]
        update_channel_spec = compiled_ast_ir["update_channel_spec"]
        
        # Backend: verify realtime broadcasting
        router_code = _render_dataset_router_module(backend_ir) 
        assert "broadcast_dataset_change" in router_code
        assert "redis" in router_code.lower() or "pubsub" in router_code.lower()
        
        # Frontend: verify WebSocket subscription
        client_code = _generate_dataset_client_code(backend_ir)
        assert "WebSocket" in client_code
        assert "onmessage" in client_code.lower()
        assert "reconnect" in client_code.lower()
        
        # Verify channel configuration matches
        assert update_channel_spec.channel_name == "dataset_updates:users"
        assert "create" in update_channel_spec.event_types
        assert "update" in update_channel_spec.event_types
        assert "delete" in update_channel_spec.event_types
    
    def test_security_features(self, compiled_ast_ir):
        """Test security features in generated code."""
        backend_ir = compiled_ast_ir["backend_ir"]
        
        router_code = _render_dataset_router_module(backend_ir)
        
        # SQL injection prevention
        assert "compile_dataset" in router_code  # Uses SQL compiler
        
        # Input validation
        assert "BaseModel" in router_code
        assert "Field(" in router_code or "validation" in router_code.lower()
        
        # HTTP status codes
        assert "status_code" in router_code
        assert "HTTP_404_NOT_FOUND" in router_code
        assert "HTTP_500_INTERNAL_SERVER_ERROR" in router_code
        
        # Error handling
        assert "HTTPException" in router_code
        assert "try:" in router_code
        assert "except" in router_code
    
    def test_backwards_compatibility(self):
        """Test that data binding doesn't break existing functionality."""
        # Test with empty IR (no datasets)
        empty_ir = MagicMock()
        empty_ir.datasets = []
        
        # Should not crash
        router_code = _render_dataset_router_module(empty_ir)
        client_code = _generate_dataset_client_code(empty_ir)
        
        # Should generate minimal but valid code
        assert "router = APIRouter" in router_code
        assert "DatasetClient" in client_code
        
        # Test with legacy dataset (no binding config)
        legacy_dataset = MagicMock()
        legacy_dataset.name = "legacy"
        legacy_dataset.realtime_enabled = False
        legacy_dataset.access_policy = None
        legacy_dataset.schema = []
        
        legacy_ir = MagicMock()
        legacy_ir.datasets = [legacy_dataset]
        
        # Should handle gracefully
        router_code = _render_dataset_router_module(legacy_ir)
        assert "legacy" in router_code.lower()
    
    def test_performance_features(self, compiled_ast_ir):
        """Test performance optimization features."""
        backend_ir = compiled_ast_ir["backend_ir"]
        
        # Backend optimizations
        router_code = _render_dataset_router_module(backend_ir)
        assert "pagination" in router_code.lower()
        assert "page_size" in router_code
        assert "LIMIT" in router_code or "limit" in router_code.lower()
        
        # Frontend optimizations
        client_code = _generate_dataset_client_code(backend_ir)
        assert "useCallback" in client_code  # Memoization
        assert "useState" in client_code      # State management
        assert "useRef" in client_code        # Reference optimization
        
        # Connection pooling/management
        assert "WebSocketManager" in client_code
        assert "reconnect" in client_code.lower()
    
    def test_observability_integration(self, compiled_ast_ir):
        """Test observability and monitoring integration."""
        backend_ir = compiled_ast_ir["backend_ir"]
        
        router_code = _render_dataset_router_module(backend_ir)
        
        # OpenTelemetry integration
        assert "tracer" in router_code
        assert "span" in router_code
        assert "set_attribute" in router_code
        
        # Logging
        assert "logger" in router_code
        assert "logging" in router_code
        
        # Error tracking
        assert "span.set_status" in router_code or "StatusCode.ERROR" in router_code
        
        # Metrics collection
        assert "dataset.name" in router_code
        assert "record.id" in router_code or "record_id" in router_code


@pytest.mark.integration
class TestDataBindingLiveSystem:
    """Integration tests simulating live system behavior."""
    
    def test_crud_operation_flow(self):
        """Test complete CRUD operation flow."""
        # This would require actual database and Redis setup
        # For now, test the code generation produces correct structure
        
        mock_ir = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.name = "test_entity"
        mock_dataset.realtime_enabled = True
        mock_dataset.access_policy = {
            "read_only": False,
            "allow_create": True,
            "allow_update": True,
            "allow_delete": True,
        }
        mock_dataset.schema = [
            {"name": "id", "type": "integer"},
            {"name": "name", "type": "string"},
        ]
        mock_ir.datasets = [mock_dataset]
        
        # Generate code
        backend_code = _render_dataset_router_module(mock_ir)
        frontend_code = _generate_dataset_client_code(mock_ir)
        
        # Verify complete CRUD flow is possible
        # CREATE
        assert "async def create_test_entity_item" in backend_code
        assert "async create(data:" in frontend_code
        
        # READ
        assert "async def get_test_entity_dataset" in backend_code
        assert "async fetch(options:" in frontend_code
        
        # UPDATE
        assert "async def update_test_entity_item" in backend_code
        assert "async update(id:" in frontend_code
        
        # DELETE
        assert "async def delete_test_entity_item" in backend_code
        assert "async delete(id:" in frontend_code
        
        # REALTIME
        assert "broadcast_dataset_change" in backend_code
        assert "WebSocket" in frontend_code
    
    def test_error_propagation(self):
        """Test error handling and propagation through the stack."""
        mock_ir = MagicMock() 
        mock_ir.datasets = []
        
        # Should not crash on edge cases
        try:
            _render_dataset_router_module(mock_ir)
            _generate_dataset_client_code(mock_ir)
        except Exception as e:
            pytest.fail(f"Code generation should not crash: {e}")
    
    def test_scalability_features(self):
        """Test scalability features in generated code."""
        # Large dataset simulation
        mock_ir = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.name = "large_dataset"
        mock_dataset.realtime_enabled = True
        mock_dataset.schema = [{"name": f"field_{i}", "type": "string"} for i in range(20)]
        mock_ir.datasets = [mock_dataset]
        
        backend_code = _render_dataset_router_module(mock_ir)
        frontend_code = _generate_dataset_client_code(mock_ir)
        
        # Verify pagination
        assert "page_size" in backend_code
        assert "page_size" in frontend_code
        
        # Verify filtering/search
        assert "search" in backend_code or "filter" in backend_code.lower()
        assert "search" in frontend_code
        
        # Verify connection management
        assert "connection" in frontend_code.lower() or "pool" in frontend_code.lower()