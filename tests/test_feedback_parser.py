"""
Tests for Feedback Component Parsing (Modal, Toast).

Tests cover:
- Basic modal and toast syntax parsing
- Nested modal content
- Modal actions with variants
- Toast variants and positioning
- Trigger-based interactions
"""

import textwrap
import pytest
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ast.pages import Modal, ModalAction, Toast, ShowText


def parse(source: str):
    """Helper to parse dedented source code."""
    return LegacyProgramParser(textwrap.dedent(source)).parse()


class TestModalParsing:
    """Test parsing of modal dialog components."""

    def test_basic_modal(self):
        """Test basic modal with minimal configuration."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "confirm":
                    title: "Confirm Action"
                    description: "Are you sure?"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None, "Modal should be parsed"
        assert modal.id == "confirm"
        assert modal.title == "Confirm Action"
        assert modal.description == "Are you sure?"
        assert modal.size == "md"  # Default size
        assert modal.dismissible is True  # Default dismissible

    def test_modal_with_size_variants(self):
        """Test modal with different size variants."""
        for size in ["sm", "md", "lg", "xl", "full"]:
            source = f'''
                app "Test"

                page "Home" at "/":
                    modal "test":
                        title: "Test"
                        size: {size}
                    
                    show text "Hello"
            '''
            module = parse(source)
            app = module.body[0]
            page = app.pages[0]
            
            modal = next((m for m in page.body if isinstance(m, Modal)), None)
            assert modal is not None
            assert modal.size == size

    def test_modal_not_dismissible(self):
        """Test modal that cannot be dismissed."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "important":
                    title: "Important"
                    dismissible: false
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert modal.dismissible is False

    def test_modal_with_trigger(self):
        """Test modal with trigger event."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "dialog":
                    title: "Dialog"
                    trigger: "open_dialog"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert modal.trigger == "open_dialog"

    def test_modal_with_nested_content(self):
        """Test modal with nested show text content."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "info":
                    title: "Information"
                    content:
                        show text "This is important information."
                        show text "Please read carefully."
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert len(modal.content) == 2
        assert isinstance(modal.content[0], ShowText)
        assert modal.content[0].text == "This is important information."
        assert isinstance(modal.content[1], ShowText)
        assert modal.content[1].text == "Please read carefully."

    def test_modal_with_single_action(self):
        """Test modal with one action button."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "alert":
                    title: "Alert"
                    actions:
                        action "OK"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert len(modal.actions) == 1
        assert modal.actions[0].label == "OK"
        assert modal.actions[0].variant == "default"
        assert modal.actions[0].close is True

    def test_modal_with_multiple_actions(self):
        """Test modal with multiple action buttons."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "confirm_delete":
                    title: "Confirm Delete"
                    actions:
                        action "Cancel" variant "ghost"
                        action "Delete" variant "destructive" action "do_delete"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert len(modal.actions) == 2
        
        # First action: Cancel
        assert modal.actions[0].label == "Cancel"
        assert modal.actions[0].variant == "ghost"
        assert modal.actions[0].close is True
        
        # Second action: Delete
        assert modal.actions[1].label == "Delete"
        assert modal.actions[1].variant == "destructive"
        assert modal.actions[1].action == "do_delete"
        assert modal.actions[1].close is True

    def test_modal_action_variants(self):
        """Test all modal action button variants."""
        variants = ["default", "primary", "destructive", "ghost", "link"]
        
        for variant in variants:
            source = f'''
                app "Test"

                page "Home" at "/":
                    modal "test":
                        title: "Test"
                        actions:
                            action "Button" variant "{variant}"
                    
                    show text "Hello"
            '''
            module = parse(source)
            app = module.body[0]
            page = app.pages[0]
            
            modal = next((m for m in page.body if isinstance(m, Modal)), None)
            assert modal is not None
            assert len(modal.actions) == 1
            assert modal.actions[0].variant == variant

    def test_modal_action_no_close(self):
        """Test modal action that doesn't close the modal."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "form":
                    title: "Form"
                    actions:
                        action "Submit" action "submit_form" close: false
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert len(modal.actions) == 1
        assert modal.actions[0].label == "Submit"
        assert modal.actions[0].action == "submit_form"
        assert modal.actions[0].close is False

    def test_modal_full_configuration(self):
        """Test modal with all features combined."""
        source = '''
            app "Test"

            page "Home" at "/":
                modal "full_modal":
                    title: "Full Configuration"
                    description: "This modal has all features"
                    size: lg
                    dismissible: true
                    trigger: "show_full_modal"
                    content:
                        show text "Modal content goes here"
                        show text "Multiple paragraphs supported"
                    actions:
                        action "Cancel" variant "ghost"
                        action "Save Draft" variant "default" action "save_draft" close: false
                        action "Publish" variant "primary" action "publish"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        modal = next((m for m in page.body if isinstance(m, Modal)), None)
        assert modal is not None
        assert modal.id == "full_modal"
        assert modal.title == "Full Configuration"
        assert modal.description == "This modal has all features"
        assert modal.size == "lg"
        assert modal.dismissible is True
        assert modal.trigger == "show_full_modal"
        assert len(modal.content) == 2
        assert len(modal.actions) == 3
        
        # Verify actions
        assert modal.actions[0].label == "Cancel"
        assert modal.actions[1].label == "Save Draft"
        assert modal.actions[1].close is False
        assert modal.actions[2].label == "Publish"
        assert modal.actions[2].variant == "primary"


class TestToastParsing:
    """Test parsing of toast notification components."""

    def test_basic_toast(self):
        """Test basic toast with minimal configuration."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "notification":
                    title: "Notification"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None, "Toast should be parsed"
        assert toast.id == "notification"
        assert toast.title == "Notification"
        assert toast.variant == "default"  # Default variant
        assert toast.duration == 3000  # Default duration
        assert toast.position == "top-right"  # Default position

    def test_toast_with_description(self):
        """Test toast with title and description."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "message":
                    title: "Success"
                    description: "Operation completed successfully"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.title == "Success"
        assert toast.description == "Operation completed successfully"

    def test_toast_variants(self):
        """Test all toast variants."""
        variants = ["default", "success", "error", "warning", "info"]
        
        for variant in variants:
            source = f'''
                app "Test"

                page "Home" at "/":
                    toast "test":
                        title: "Test"
                        variant: {variant}
                    
                    show text "Hello"
            '''
            module = parse(source)
            app = module.body[0]
            page = app.pages[0]
            
            toast = next((t for t in page.body if isinstance(t, Toast)), None)
            assert toast is not None
            assert toast.variant == variant

    def test_toast_custom_duration(self):
        """Test toast with custom duration."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "long_message":
                    title: "Long Message"
                    duration: 5000
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.duration == 5000

    def test_toast_no_auto_dismiss(self):
        """Test toast that doesn't auto-dismiss."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "persistent":
                    title: "Persistent"
                    duration: 0
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.duration == 0

    def test_toast_positions(self):
        """Test all toast positioning options."""
        positions = ["top", "top-right", "top-left", "bottom", "bottom-right", "bottom-left"]
        
        for position in positions:
            source = f'''
                app "Test"

                page "Home" at "/":
                    toast "test":
                        title: "Test"
                        position: {position}
                    
                    show text "Hello"
            '''
            module = parse(source)
            app = module.body[0]
            page = app.pages[0]
            
            toast = next((t for t in page.body if isinstance(t, Toast)), None)
            assert toast is not None
            assert toast.position == position

    def test_toast_with_action(self):
        """Test toast with action button."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "undo_toast":
                    title: "Item Deleted"
                    action_label: "Undo"
                    action: "undo_delete"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.action_label == "Undo"
        assert toast.action == "undo_delete"

    def test_toast_with_trigger(self):
        """Test toast with trigger event."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "event_toast":
                    title: "Event Triggered"
                    trigger: "show_notification"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.trigger == "show_notification"

    def test_toast_full_configuration(self):
        """Test toast with all features combined."""
        source = '''
            app "Test"

            page "Home" at "/":
                toast "full_toast":
                    title: "Full Configuration"
                    description: "This toast has all features"
                    variant: warning
                    duration: 4000
                    position: bottom-right
                    action_label: "View Details"
                    action: "view_details"
                    trigger: "show_full_toast"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        toast = next((t for t in page.body if isinstance(t, Toast)), None)
        assert toast is not None
        assert toast.id == "full_toast"
        assert toast.title == "Full Configuration"
        assert toast.description == "This toast has all features"
        assert toast.variant == "warning"
        assert toast.duration == 4000
        assert toast.position == "bottom-right"
        assert toast.action_label == "View Details"
        assert toast.action == "view_details"
        assert toast.trigger == "show_full_toast"
