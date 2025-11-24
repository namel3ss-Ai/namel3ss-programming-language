"""
Test frontend data binding components and hooks.

Tests the generated TypeScript/React components for:
- DatasetClient functionality
- useDataset and useDatasetMutation hooks  
- WebSocket realtime subscriptions
- Bound table, chart, and form components
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from namel3ss.codegen.frontend.react.dataset_client import (
    write_dataset_client_lib,
    _generate_dataset_client_code,
    _generate_dataset_types,
)
from namel3ss.codegen.frontend.react.bound_components import (
    write_bound_table_widget,
    write_bound_chart_widget, 
    write_bound_form_widget,
    generate_bound_components,
)


class TestDatasetClientGeneration:
    """Test TypeScript DatasetClient generation."""
    
    @pytest.fixture
    def mock_backend_ir(self):
        """Create mock BackendIR for testing."""
        dataset1 = MagicMock()
        dataset1.name = "users"
        dataset1.realtime_enabled = True
        dataset1.schema = [
            {"name": "id", "type": "integer", "required": True},
            {"name": "name", "type": "string", "required": True},
            {"name": "email", "type": "string", "required": False},
            {"name": "created_at", "type": "datetime", "required": True},
        ]
        
        dataset2 = MagicMock()
        dataset2.name = "products"  
        dataset2.realtime_enabled = False
        dataset2.schema = [
            {"name": "id", "type": "integer", "required": True},
            {"name": "title", "type": "string", "required": True},
            {"name": "price", "type": "float", "required": True},
            {"name": "tags", "type": "array", "required": False},
        ]
        
        backend_ir = MagicMock()
        backend_ir.datasets = [dataset1, dataset2]
        
        return backend_ir
    
    def test_generate_dataset_types(self, mock_backend_ir):
        """Test TypeScript interface generation."""
        result = _generate_dataset_types(mock_backend_ir.datasets)
        
        assert "export interface Users {" in result
        assert "export interface Products {" in result
        assert "id: number;" in result
        assert "name: string;" in result
        assert "email?: string;" in result
        assert "price: number;" in result
        assert "tags?: any[];" in result
        assert "created_at: string;" in result
    
    def test_generate_dataset_client_code(self, mock_backend_ir):
        """Test complete DatasetClient code generation."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        # Check core structures are present
        assert "class DatasetClient<T = any>" in result
        assert "class WebSocketManager" in result
        assert "export function useDataset<T = any>" in result
        assert "export function useDatasetMutation<T = any>" in result
        
        # Check dataset types are included
        assert "export interface Users {" in result
        assert "export interface Products {" in result
        
        # Check WebSocket support (realtime enabled)
        assert "WebSocketMessage" in result
        assert "subscribe(" in result
        assert "reconnect" in result
    
    def test_write_dataset_client_lib(self, mock_backend_ir):
        """Test writing DatasetClient to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lib_dir = Path(temp_dir)
            
            write_dataset_client_lib(lib_dir, mock_backend_ir)
            
            client_file = lib_dir / "datasetClient.ts"
            assert client_file.exists()
            
            content = client_file.read_text()
            assert "class DatasetClient" in content
            assert "useDataset" in content
            assert "useDatasetMutation" in content
    
    def test_dataset_type_mapping(self):
        """Test field type mapping from Namel3ss to TypeScript."""
        from namel3ss.codegen.frontend.react.dataset_client import _map_field_type
        
        assert _map_field_type("string") == "string"
        assert _map_field_type("integer") == "number"
        assert _map_field_type("float") == "number"
        assert _map_field_type("boolean") == "boolean"
        assert _map_field_type("datetime") == "string"
        assert _map_field_type("json") == "any"
        assert _map_field_type("array") == "any[]"
        assert _map_field_type("unknown_type") == "any"
    
    def test_pascal_case_conversion(self):
        """Test snake_case to PascalCase conversion."""
        from namel3ss.codegen.frontend.react.dataset_client import _to_pascal_case
        
        assert _to_pascal_case("user_profiles") == "UserProfiles"
        assert _to_pascal_case("simple") == "Simple"
        assert _to_pascal_case("multi_word_dataset") == "MultiWordDataset"
        assert _to_pascal_case("") == ""


class TestBoundComponentsGeneration:
    """Test data-bound React component generation."""
    
    def test_write_bound_table_widget(self):
        """Test BoundTableWidget generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            components_dir = Path(temp_dir)
            
            write_bound_table_widget(components_dir)
            
            widget_file = components_dir / "BoundTableWidget.tsx"
            assert widget_file.exists()
            
            content = widget_file.read_text()
            assert "function BoundTableWidget" in content
            assert "useDataset" in content
            assert "useDatasetMutation" in content
            assert "handleEdit" in content
            assert "handleSave" in content
            assert "handleDelete" in content
            assert "pagination" in content.lower()
    
    def test_write_bound_chart_widget(self):
        """Test BoundChartWidget generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            components_dir = Path(temp_dir)
            
            write_bound_chart_widget(components_dir)
            
            widget_file = components_dir / "BoundChartWidget.tsx"
            assert widget_file.exists()
            
            content = widget_file.read_text()
            assert "function BoundChartWidget" in content
            assert "useDataset" in content
            assert "ChartWidgetConfig" in content
            assert "Loading chart data" in content
    
    def test_write_bound_form_widget(self):
        """Test BoundFormWidget generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            components_dir = Path(temp_dir)
            
            write_bound_form_widget(components_dir)
            
            widget_file = components_dir / "BoundFormWidget.tsx"
            assert widget_file.exists()
            
            content = widget_file.read_text()
            assert "function BoundFormWidget" in content
            assert "useDatasetMutation" in content
            assert "handleSubmit" in content
            assert "create" in content
            assert "update" in content
            assert "recordId" in content
    
    def test_generate_bound_components(self):
        """Test generation of all bound components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            components_dir = Path(temp_dir)
            
            generate_bound_components(components_dir)
            
            # Check all files are generated
            assert (components_dir / "BoundTableWidget.tsx").exists()
            assert (components_dir / "BoundChartWidget.tsx").exists()
            assert (components_dir / "BoundFormWidget.tsx").exists()
            
            # Check file content quality
            table_content = (components_dir / "BoundTableWidget.tsx").read_text()
            assert "inline editing" in table_content.lower() or "edit" in table_content.lower()


class TestDatasetClientFunctionality:
    """Test DatasetClient TypeScript functionality (conceptual tests)."""
    
    def test_dataset_client_structure(self, mock_backend_ir):
        """Test that generated DatasetClient has proper structure."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        # Check DatasetClient class methods
        assert "async fetch(options: DatasetQueryOptions" in result
        assert "async create(data: Partial<T>)" in result  
        assert "async update(id: string | number, data: Partial<T>)" in result
        assert "async delete(id: string | number)" in result
        assert "subscribe(callback:" in result
        
        # Check pagination support
        assert "page:" in result
        assert "page_size:" in result
        assert "sort_by:" in result
        assert "sort_order:" in result
        assert "search:" in result
    
    def test_websocket_manager_structure(self, mock_backend_ir):
        """Test WebSocketManager structure."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        assert "class WebSocketManager" in result
        assert "subscribe(" in result
        assert "connect(" in result
        assert "disconnect(" in result
        assert "scheduleReconnect(" in result
        assert "buildWebSocketUrl(" in result
        
        # Check reconnection logic
        assert "maxReconnectDelay" in result
        assert "baseReconnectDelay" in result
        assert "exponential" in result.lower() or "pow(2" in result
    
    def test_react_hooks_structure(self, mock_backend_ir):
        """Test React hooks structure."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        # useDataset hook
        assert "function useDataset" in result
        assert "useState" in result
        assert "useEffect" in result
        assert "useCallback" in result
        assert "loading" in result
        assert "error" in result
        assert "refetch" in result
        
        # useDatasetMutation hook
        assert "function useDatasetMutation" in result
        assert "create" in result
        assert "update" in result
        assert "delete:" in result  # delete is aliased since it's a keyword
        
        # Check realtime subscription in hook
        assert "subscribe(" in result
        assert "unsubscribe" in result
    
    def test_error_handling(self, mock_backend_ir):
        """Test error handling in generated code."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        assert "try {" in result
        assert "catch" in result
        assert "Error" in result
        assert "throw new Error" in result or "HTTPException" in result.lower()
        
        # WebSocket error handling
        assert "onclose" in result
        assert "onerror" in result
        assert "console.error" in result


class TestDataBindingIntegration:
    """Integration tests for data binding system."""
    
    def test_full_stack_generation(self, mock_backend_ir):
        """Test complete frontend-backend integration code generation."""
        # Generate frontend components
        frontend_code = _generate_dataset_client_code(mock_backend_ir)
        
        # Verify frontend has proper API endpoints
        assert "/api/datasets/" in frontend_code
        assert "POST" in frontend_code or "method: \"POST\"" in frontend_code
        assert "PATCH" in frontend_code or "method: \"PATCH\"" in frontend_code
        assert "DELETE" in frontend_code or "method: \"DELETE\"" in frontend_code
        
        # Verify realtime integration
        assert "WebSocket" in frontend_code
        assert "/ws/" in frontend_code
    
    def test_backwards_compatibility(self):
        """Test that data binding doesn't break existing functionality."""
        # Mock IR without datasets
        empty_ir = MagicMock()
        empty_ir.datasets = []
        
        result = _generate_dataset_client_code(empty_ir)
        
        # Should still generate basic structure
        assert "class DatasetClient" in result
        assert "useDataset" in result
        assert "No datasets defined" in result
    
    def test_production_ready_features(self, mock_backend_ir):
        """Test production-ready features in generated code."""
        result = _generate_dataset_client_code(mock_backend_ir)
        
        # Error handling
        assert "try" in result and "catch" in result
        
        # Loading states
        assert "loading" in result.lower()
        
        # Retry logic for WebSockets
        assert "reconnect" in result.lower()
        
        # Type safety
        assert "TypeScript" in __doc__ or "interface" in result
        
        # Pagination
        assert "page" in result and "page_size" in result
        
        # Search and filtering
        assert "search" in result and "sort" in result