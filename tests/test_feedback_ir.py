"""
Tests for Feedback Component IR Building (Modal, Toast).

Tests cover:
- AST to IR conversion for modals and toasts
- Nested modal content conversion
- Modal action conversion
- Toast property mapping
"""

import pytest
from namel3ss.ast.pages import Modal, ModalAction, Toast, ShowText
from namel3ss.ir.builder import _statement_to_component_spec
from namel3ss.ir.spec import IRModal, IRModalAction, IRToast, ComponentSpec


def create_state():
    """Create a minimal builder state for testing."""
    return {}


class TestModalIRBuilding:
    """Test Modal AST → IR conversion."""

    def test_basic_modal_conversion(self):
        """Test basic modal converts to ComponentSpec."""
        modal = Modal(
            id="test_modal",
            title="Test Modal",
            description="Test description"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        assert spec is not None
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "test_modal"
        assert spec.type == "modal"
        assert spec.props["id"] == "test_modal"
        assert spec.props["title"] == "Test Modal"
        assert spec.props["description"] == "Test description"

    def test_modal_size_variants(self):
        """Test modal size variants are preserved."""
        for size in ["sm", "md", "lg", "xl", "full"]:
            modal = Modal(
                id="test",
                title="Test",
                size=size
            )
            
            state = create_state()
            spec = _statement_to_component_spec(modal, state)
            
            assert spec.props["size"] == size

    def test_modal_dismissible_flag(self):
        """Test modal dismissible property conversion."""
        # Dismissible modal
        modal_dismissible = Modal(
            id="test1",
            title="Test",
            dismissible=True
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal_dismissible, state)
        assert spec.props["dismissible"] is True
        
        # Non-dismissible modal
        modal_not_dismissible = Modal(
            id="test2",
            title="Test",
            dismissible=False
        )
        
        spec = _statement_to_component_spec(modal_not_dismissible, state)
        assert spec.props["dismissible"] is False

    def test_modal_trigger(self):
        """Test modal trigger property conversion."""
        modal = Modal(
            id="test",
            title="Test",
            trigger="open_modal"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        assert spec.props["trigger"] == "open_modal"

    def test_modal_with_nested_content(self):
        """Test modal with nested content converts to children."""
        modal = Modal(
            id="test",
            title="Test",
            content=[
                ShowText(text="First paragraph"),
                ShowText(text="Second paragraph")
            ]
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        assert len(spec.children) == 2
        assert spec.children[0].type == "text"
        assert spec.children[1].type == "text"

    def test_modal_with_actions(self):
        """Test modal actions conversion to IR."""
        modal = Modal(
            id="test",
            title="Test",
            actions=[
                ModalAction(label="Cancel", variant="ghost"),
                ModalAction(label="Confirm", variant="primary", action="confirm_action")
            ]
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        # Check IR spec in metadata
        ir_modal = spec.metadata.get("ir_spec")
        assert ir_modal is not None
        assert isinstance(ir_modal, IRModal)
        assert len(ir_modal.actions) == 2
        
        # First action
        assert ir_modal.actions[0].label == "Cancel"
        assert ir_modal.actions[0].variant == "ghost"
        assert ir_modal.actions[0].close is True
        
        # Second action
        assert ir_modal.actions[1].label == "Confirm"
        assert ir_modal.actions[1].variant == "primary"
        assert ir_modal.actions[1].action == "confirm_action"

    def test_modal_action_variants(self):
        """Test all modal action variants convert correctly."""
        variants = ["default", "primary", "destructive", "ghost", "link"]
        
        for variant in variants:
            modal = Modal(
                id="test",
                title="Test",
                actions=[ModalAction(label="Button", variant=variant)]
            )
            
            state = create_state()
            spec = _statement_to_component_spec(modal, state)
            
            ir_modal = spec.metadata.get("ir_spec")
            assert ir_modal.actions[0].variant == variant

    def test_modal_action_close_behavior(self):
        """Test modal action close property."""
        modal = Modal(
            id="test",
            title="Test",
            actions=[
                ModalAction(label="Submit", action="submit", close=False),
                ModalAction(label="Close", close=True)
            ]
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        ir_modal = spec.metadata.get("ir_spec")
        assert ir_modal.actions[0].close is False
        assert ir_modal.actions[1].close is True

    def test_modal_full_configuration(self):
        """Test modal with all features converts correctly."""
        modal = Modal(
            id="full_modal",
            title="Full Modal",
            description="Complete configuration",
            size="lg",
            dismissible=True,
            trigger="show_modal",
            content=[ShowText(text="Content")],
            actions=[
                ModalAction(label="Cancel", variant="ghost"),
                ModalAction(label="Save", variant="primary", action="save")
            ]
        )
        
        state = create_state()
        spec = _statement_to_component_spec(modal, state)
        
        assert spec.name == "full_modal"
        assert spec.type == "modal"
        assert spec.props["title"] == "Full Modal"
        assert spec.props["description"] == "Complete configuration"
        assert spec.props["size"] == "lg"
        assert spec.props["dismissible"] is True
        assert spec.props["trigger"] == "show_modal"
        assert len(spec.children) == 1
        
        ir_modal = spec.metadata.get("ir_spec")
        assert len(ir_modal.actions) == 2


class TestToastIRBuilding:
    """Test Toast AST → IR conversion."""

    def test_basic_toast_conversion(self):
        """Test basic toast converts to ComponentSpec."""
        toast = Toast(
            id="test_toast",
            title="Test Toast",
            description="Test description"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec is not None
        assert isinstance(spec, ComponentSpec)
        assert spec.name == "test_toast"
        assert spec.type == "toast"
        assert spec.props["id"] == "test_toast"
        assert spec.props["title"] == "Test Toast"
        assert spec.props["description"] == "Test description"

    def test_toast_variants(self):
        """Test all toast variants convert correctly."""
        variants = ["default", "success", "error", "warning", "info"]
        
        for variant in variants:
            toast = Toast(
                id="test",
                title="Test",
                variant=variant
            )
            
            state = create_state()
            spec = _statement_to_component_spec(toast, state)
            
            assert spec.props["variant"] == variant

    def test_toast_duration(self):
        """Test toast duration property conversion."""
        toast = Toast(
            id="test",
            title="Test",
            duration=5000
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec.props["duration"] == 5000

    def test_toast_no_auto_dismiss(self):
        """Test toast with duration 0 (no auto-dismiss)."""
        toast = Toast(
            id="test",
            title="Test",
            duration=0
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec.props["duration"] == 0

    def test_toast_positions(self):
        """Test all toast positions convert correctly."""
        positions = ["top", "top-right", "top-left", "bottom", "bottom-right", "bottom-left"]
        
        for position in positions:
            toast = Toast(
                id="test",
                title="Test",
                position=position
            )
            
            state = create_state()
            spec = _statement_to_component_spec(toast, state)
            
            assert spec.props["position"] == position

    def test_toast_with_action(self):
        """Test toast with action button converts correctly."""
        toast = Toast(
            id="test",
            title="Test",
            action_label="Undo",
            action="undo_action"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec.props["action_label"] == "Undo"
        assert spec.props["action"] == "undo_action"

    def test_toast_trigger(self):
        """Test toast trigger property conversion."""
        toast = Toast(
            id="test",
            title="Test",
            trigger="show_toast"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec.props["trigger"] == "show_toast"

    def test_toast_ir_spec(self):
        """Test toast IR spec is created in metadata."""
        toast = Toast(
            id="test",
            title="Test Toast",
            description="Description",
            variant="success"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        ir_toast = spec.metadata.get("ir_spec")
        assert ir_toast is not None
        assert isinstance(ir_toast, IRToast)
        assert ir_toast.id == "test"
        assert ir_toast.title == "Test Toast"
        assert ir_toast.description == "Description"
        assert ir_toast.variant == "success"

    def test_toast_full_configuration(self):
        """Test toast with all features converts correctly."""
        toast = Toast(
            id="full_toast",
            title="Full Toast",
            description="Complete configuration",
            variant="warning",
            duration=4000,
            position="bottom-right",
            action_label="View",
            action="view_details",
            trigger="show_toast"
        )
        
        state = create_state()
        spec = _statement_to_component_spec(toast, state)
        
        assert spec.name == "full_toast"
        assert spec.type == "toast"
        assert spec.props["title"] == "Full Toast"
        assert spec.props["description"] == "Complete configuration"
        assert spec.props["variant"] == "warning"
        assert spec.props["duration"] == 4000
        assert spec.props["position"] == "bottom-right"
        assert spec.props["action_label"] == "View"
        assert spec.props["action"] == "view_details"
        assert spec.props["trigger"] == "show_toast"
        
        ir_toast = spec.metadata.get("ir_spec")
        assert ir_toast.variant == "warning"
        assert ir_toast.duration == 4000
        assert ir_toast.position == "bottom-right"
