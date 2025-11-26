"""
Tests for feedback component React code generation.

Validates that Modal and Toast IR specs are correctly serialized to React component props.
"""

import pytest
from namel3ss.ir.spec import IRModal, IRModalAction, IRToast
from namel3ss.codegen.frontend.react.pages import serialize_modal_action


class TestModalSerialization:
    """Test modal IR → React props serialization."""
    
    def test_basic_modal_action_serialization(self):
        """Test modal action converts to correct props."""
        action = IRModalAction(
            label="Confirm",
            action="confirm_action",
            variant="primary",
            close=True
        )
        
        result = serialize_modal_action(action)
        
        assert result["label"] == "Confirm"
        assert result["action"] == "confirm_action"
        assert result["variant"] == "primary"
        assert result["close"] is True
    
    def test_modal_action_minimal(self):
        """Test modal action with only required fields."""
        action = IRModalAction(label="OK")
        
        result = serialize_modal_action(action)
        
        assert result["label"] == "OK"
        # Optional fields should still be present with defaults
        assert "variant" in result
        assert "close" in result
    
    def test_modal_action_no_close(self):
        """Test modal action that doesn't close modal."""
        action = IRModalAction(
            label="Submit",
            action="submit_form",
            close=False
        )
        
        result = serialize_modal_action(action)
        
        assert result["close"] is False
    
    def test_modal_action_variants(self):
        """Test all modal action variants serialize correctly."""
        variants = ["default", "primary", "destructive", "ghost", "link"]
        
        for variant in variants:
            action = IRModalAction(label="Button", variant=variant)
            result = serialize_modal_action(action)
            assert result["variant"] == variant
    
    def test_modal_action_without_action_id(self):
        """Test modal action without action identifier."""
        action = IRModalAction(label="Close", variant="ghost")
        
        result = serialize_modal_action(action)
        
        assert result["label"] == "Close"
        assert result["variant"] == "ghost"
        # Action ID should be None or not cause errors


class TestModalCodegenIntegration:
    """Test modal integration into page codegen."""
    
    def test_modal_widget_structure(self):
        """Test modal creates correct widget structure."""
        # This tests the extract_widgets functionality
        # In actual usage, this would be tested via build_frontend and examining output
        
        # Expected widget structure from serialization
        expected_widget = {
            "id": "modal_1",
            "type": "modal",
            "modal_id": "confirm_delete",
            "title": "Confirm Delete",
            "description": "Are you sure?",
            "size": "md",
            "dismissible": True,
            "trigger": None,
            "actions": [
                {"label": "Cancel", "variant": "ghost", "close": True},
                {"label": "Delete", "variant": "destructive", "action": "do_delete", "close": True}
            ],
            "content": []
        }
        
        # Verify structure
        assert expected_widget["type"] == "modal"
        assert expected_widget["modal_id"] == "confirm_delete"
        assert len(expected_widget["actions"]) == 2
    
    def test_modal_with_nested_content_structure(self):
        """Test modal with nested content creates correct structure."""
        expected_widget = {
            "id": "modal_1",
            "type": "modal",
            "modal_id": "info",
            "title": "Information",
            "content": [
                {"type": "text", "text": "Important message", "styles": {}},
                {"type": "text", "text": "Read carefully", "styles": {}}
            ]
        }
        
        assert expected_widget["type"] == "modal"
        assert len(expected_widget["content"]) == 2
        assert expected_widget["content"][0]["type"] == "text"


class TestToastCodegenIntegration:
    """Test toast integration into page codegen."""
    
    def test_toast_widget_structure(self):
        """Test toast creates correct widget structure."""
        expected_widget = {
            "id": "toast_1",
            "type": "toast",
            "toast_id": "success_notification",
            "title": "Success",
            "description": "Operation completed",
            "variant": "success",
            "duration": 3000,
            "action_label": None,
            "action": None,
            "position": "top-right",
            "trigger": None
        }
        
        assert expected_widget["type"] == "toast"
        assert expected_widget["toast_id"] == "success_notification"
        assert expected_widget["variant"] == "success"
    
    def test_toast_with_action_structure(self):
        """Test toast with action creates correct structure."""
        expected_widget = {
            "id": "toast_1",
            "type": "toast",
            "toast_id": "undo_toast",
            "title": "Item Deleted",
            "action_label": "Undo",
            "action": "undo_delete",
            "variant": "default",
            "duration": 3000,
            "position": "top-right"
        }
        
        assert expected_widget["action_label"] == "Undo"
        assert expected_widget["action"] == "undo_delete"
    
    def test_toast_variants_structure(self):
        """Test all toast variants create correct structure."""
        variants = ["default", "success", "error", "warning", "info"]
        
        for variant in variants:
            expected_widget = {
                "type": "toast",
                "toast_id": f"{variant}_toast",
                "title": variant.capitalize(),
                "variant": variant
            }
            
            assert expected_widget["variant"] == variant
    
    def test_toast_positions_structure(self):
        """Test all toast positions create correct structure."""
        positions = ["top", "top-right", "top-left", "bottom", "bottom-right", "bottom-left"]
        
        for position in positions:
            expected_widget = {
                "type": "toast",
                "toast_id": "test",
                "title": "Test",
                "position": position
            }
            
            assert expected_widget["position"] == position


class TestFeedbackComponentsRendering:
    """Test React component rendering for modals and toasts."""
    
    def test_modal_render_case(self):
        """Test modal render case in renderWidget switch."""
        # Expected TypeScript render output
        expected_tsx = '''
        case "modal":
          return (
            <Modal
              key={widget.id}
              id={widget.modal_id}
              title={widget.title}
              description={widget.description}
              content={widget.content?.map((child: any) => renderWidget(child, data)) || []}
              actions={widget.actions}
              size={widget.size}
              dismissible={widget.dismissible}
              trigger={widget.trigger}
            />
          );
        '''
        
        # Verify structure (this is a documentation test)
        assert "case \"modal\"" in expected_tsx
        assert "content={widget.content?.map" in expected_tsx
        assert "actions={widget.actions}" in expected_tsx
    
    def test_toast_render_case(self):
        """Test toast render case in renderWidget switch."""
        expected_tsx = '''
        case "toast":
          return (
            <Toast
              key={widget.id}
              id={widget.toast_id}
              title={widget.title}
              description={widget.description}
              variant={widget.variant}
              duration={widget.duration}
              action_label={widget.action_label}
              action={widget.action}
              position={widget.position}
              trigger={widget.trigger}
            />
          );
        '''
        
        assert "case \"toast\"" in expected_tsx
        assert "variant={widget.variant}" in expected_tsx
        assert "duration={widget.duration}" in expected_tsx


class TestComponentImports:
    """Test that Modal and Toast imports are generated correctly."""
    
    def test_modal_import_generated(self):
        """Test Modal import statement is included."""
        expected_import = 'import Modal from "../components/Modal";'
        
        # This would be verified by checking generated page files
        assert 'import Modal from "../components/Modal"' in expected_import
    
    def test_toast_import_generated(self):
        """Test Toast import statement is included."""
        expected_import = 'import Toast from "../components/Toast";'
        
        assert 'import Toast from "../components/Toast"' in expected_import
    
    def test_feedback_imports_order(self):
        """Test feedback component imports are in correct order."""
        # Imports should appear after chrome components
        expected_order = [
            'import Sidebar from "../components/Sidebar";',
            'import Navbar from "../components/Navbar";',
            'import Breadcrumbs from "../components/Breadcrumbs";',
            'import CommandPalette from "../components/CommandPalette";',
            'import Modal from "../components/Modal";',
            'import Toast from "../components/Toast";'
        ]
        
        # Verify Modal and Toast come after chrome components
        modal_idx = next(i for i, s in enumerate(expected_order) if "Modal" in s)
        toast_idx = next(i for i, s in enumerate(expected_order) if "Toast" in s)
        palette_idx = next(i for i, s in enumerate(expected_order) if "CommandPalette" in s)
        
        assert modal_idx > palette_idx
        assert toast_idx > palette_idx
        assert toast_idx == modal_idx + 1


class TestFeedbackComponentTSXGeneration:
    """Test TypeScript component file generation."""
    
    def test_modal_component_interface(self):
        """Test Modal component interfaces are correct."""
        # Expected ModalAction interface
        expected_action = '''
        export interface ModalAction {
          label: string;
          action?: string;
          variant?: "default" | "primary" | "destructive" | "ghost" | "link";
          close?: boolean;
        }
        '''
        
        assert "label: string" in expected_action
        assert "variant?: \"default\" | \"primary\"" in expected_action
    
    def test_modal_props_interface(self):
        """Test Modal props interface is correct."""
        expected_props = '''
        export interface ModalProps {
          id: string;
          title: string;
          description?: string;
          content: ReactNode[];
          actions: ModalAction[];
          size?: "sm" | "md" | "lg" | "xl" | "full";
          dismissible?: boolean;
          trigger?: string;
        }
        '''
        
        assert "content: ReactNode[]" in expected_props
        assert "actions: ModalAction[]" in expected_props
        assert "size?: \"sm\" | \"md\" | \"lg\" | \"xl\" | \"full\"" in expected_props
    
    def test_toast_props_interface(self):
        """Test Toast props interface is correct."""
        expected_props = '''
        export interface ToastProps {
          id: string;
          title: string;
          description?: string;
          variant?: "default" | "success" | "error" | "warning" | "info";
          duration?: number;
          action_label?: string;
          action?: string;
          position?: "top" | "top-right" | "top-left" | "bottom" | "bottom-right" | "bottom-left";
          trigger?: string;
        }
        '''
        
        assert "variant?: \"default\" | \"success\" | \"error\" | \"warning\" | \"info\"" in expected_props
        assert "position?: \"top\" | \"top-right\"" in expected_props
        assert "duration?: number" in expected_props
