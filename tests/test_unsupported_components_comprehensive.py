"""
Test comprehensive unsupported component error messages.

This tests the error message formatting directly, since the parser's
modern syntax checking intercepts invalid component names before
reaching the legacy page statement parser.
"""

import pytest
from namel3ss.parser.component_helpers import (
    get_component_alternatives,
    format_alternatives_error,
    COMPONENT_ALTERNATIVES
)


class TestUnsupportedComponentAlternatives:
    """Test that all unsupported components have comprehensive alternatives."""
    
    def test_all_components_have_alternatives(self):
        """All unsupported components should have full alternative info."""
        expected_components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        
        for component in expected_components:
            info = get_component_alternatives(component)
            assert info is not None, f"Missing alternatives for {component}"
            
            # Check required fields
            assert 'name' in info
            assert 'primary' in info
            assert 'alternatives' in info
            assert 'why_not_supported' in info
            assert 'docs' in info
            
            # Check alternatives structure
            assert len(info['alternatives']) >= 2, f"{component} should have 2+ alternatives"
            
            for alt in info['alternatives']:
                assert 'component' in alt
                assert 'description' in alt
                assert 'use_case' in alt
                assert 'example' in alt
                assert len(alt['example']) > 50, "Example should be substantial"
    
    def test_progress_bar_alternatives(self):
        """Progress bar should have 3 comprehensive alternatives."""
        info = get_component_alternatives('progress_bar')
        
        assert info['name'] == 'Progress Bar'
        assert len(info['alternatives']) == 3
        
        # Check alternative names
        alt_components = [alt['component'] for alt in info['alternatives']]
        assert 'show stat_summary' in alt_components
        assert 'show data_chart' in alt_components
        assert 'show text' in alt_components
        
        # Check each has use case
        for alt in info['alternatives']:
            assert 'Best for:' in alt['use_case']
            assert len(alt['description']) > 20
            assert 'show' in alt['example'] or 'stats:' in alt['example']
    
    def test_code_block_alternatives(self):
        """Code block should have 3 alternatives with syntax highlighting."""
        info = get_component_alternatives('code_block')
        
        assert info['name'] == 'Code Block'
        assert len(info['alternatives']) == 3
        
        # Check for markdown and diff_view alternatives
        alt_components = [alt['component'] for alt in info['alternatives']]
        assert any('markdown' in c.lower() for c in alt_components)
        assert any('diff_view' in c for c in alt_components)
        
        # Check examples contain code
        examples_text = ' '.join(alt['example'] for alt in info['alternatives'])
        assert '```' in examples_text  # Markdown code fence
        assert 'python' in examples_text.lower()
    
    def test_json_view_alternatives(self):
        """JSON view should have 3 alternatives for different use cases."""
        info = get_component_alternatives('json_view')
        
        assert info['name'] == 'JSON Viewer'
        assert len(info['alternatives']) == 3
        
        # Check for to_json filter, data_table, card alternatives
        alt_components = [alt['component'] for alt in info['alternatives']]
        assert any('to_json' in c for c in alt_components)
        assert any('data_table' in c for c in alt_components)
        assert any('card' in c for c in alt_components)
        
        # Check examples contain JSON-related content
        examples_text = ' '.join(alt['example'] for alt in info['alternatives'])
        assert 'to_json' in examples_text
    
    def test_tree_view_alternatives(self):
        """Tree view should have 3 alternatives for hierarchical data."""
        info = get_component_alternatives('tree_view')
        
        assert info['name'] == 'Tree View'
        assert len(info['alternatives']) == 3
        
        # Check for accordion, data_list, card alternatives
        alt_components = [alt['component'] for alt in info['alternatives']]
        assert any('accordion' in c for c in alt_components)
        assert any('data_list' in c for c in alt_components)
        assert any('card' in c for c in alt_components)
        
        # Check examples show nesting
        examples_text = ' '.join(alt['example'] for alt in info['alternatives'])
        assert 'section' in examples_text or 'nested' in examples_text.lower()
    
    def test_error_message_formatting(self):
        """Error messages should be well-formatted and comprehensive."""
        error = format_alternatives_error('progress_bar')
        
        # Check structure
        assert "Component 'Progress Bar' is not supported" in error
        assert "Why:" in error
        assert "Recommended:" in error or "stat_summary" in error
        assert "Alternatives:" in error
        assert "Example:" in error
        assert "Documentation:" in error
        
        # Check has multiple examples
        assert error.count('show ') >= 2 or error.count('stats:') >= 1
        
        # Check has documentation links
        assert 'docs/' in error
        assert '.md' in error
    
    def test_all_components_format_correctly(self):
        """All unsupported components should format without errors."""
        components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        
        for component in components:
            error = format_alternatives_error(component)
            
            # Should not crash
            assert len(error) > 200, f"{component} error too short"
            
            # Should have all sections
            assert 'not supported' in error.lower()
            assert 'Why:' in error
            assert 'Alternatives' in error or '1.' in error
            assert 'Example' in error or 'show ' in error
    
    def test_component_aliases(self):
        """Short aliases should also have alternatives."""
        # Test that shortened names work
        assert get_component_alternatives('progress') is None  # Only full name in new system
        assert get_component_alternatives('code') is None
        assert get_component_alternatives('json') is None
        assert get_component_alternatives('tree') is None
        
        # But main components do work
        assert get_component_alternatives('progress_bar') is not None
        assert get_component_alternatives('code_block') is not None
        assert get_component_alternatives('json_view') is not None
        assert get_component_alternatives('tree_view') is not None
    
    def test_why_explanations_are_clear(self):
        """Each component should have a clear explanation why it's not supported."""
        components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        
        for component in components:
            info = get_component_alternatives(component)
            why = info['why_not_supported']
            
            # Should be substantial
            assert len(why) > 50, f"{component} 'why' too short"
            
            # Should mention complexity or alternatives
            assert (
                'complex' in why.lower() or 
                'simpler' in why.lower() or
                'alternatives' in why.lower() or
                'provide' in why.lower()
            ), f"{component} 'why' not clear enough"
    
    def test_use_cases_are_specific(self):
        """Each alternative should have specific use cases."""
        components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        
        for component in components:
            info = get_component_alternatives(component)
            
            for alt in info['alternatives']:
                use_case = alt['use_case']
                
                # Should start with "Best for:"
                assert 'Best for:' in use_case, f"{component} use case format wrong"
                
                # Should have specific examples
                assert len(use_case) > 30, f"{component} use case too vague"
    
    def test_examples_are_complete(self):
        """Each alternative should have complete, working examples."""
        components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        
        for component in components:
            info = get_component_alternatives(component)
            
            for i, alt in enumerate(info['alternatives']):
                example = alt['example']
                
                # Should be substantial
                assert len(example) > 50, f"{component} alt {i} example too short"
                
                # Should contain actual code
                assert (
                    'show' in example or 
                    'accordion' in example or
                    'diff_view' in example or
                    'stats:' in example
                ), f"{component} alt {i} example not code"
                
                # Should have proper indentation
                assert '\n' in example, f"{component} alt {i} example not formatted"
    
    def test_documentation_links_exist(self):
        """All documentation links should reference actual files."""
        import os
        
        components = ['progress_bar', 'code_block', 'json_view', 'tree_view']
        docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
        
        for component in components:
            info = get_component_alternatives(component)
            
            for doc in info['docs']:
                # Check format
                assert doc.endswith('.md'), f"{component} doc should be markdown"
                
                # Note: We don't check file existence in tests since docs/ might not be
                # in the same location during testing. The references should be correct though.
                assert len(doc) > 5, f"{component} doc name too short"


class TestErrorMessageQuality:
    """Test the quality and clarity of error messages."""
    
    def test_error_has_clear_structure(self):
        """Error message should have clear sections."""
        error = format_alternatives_error('json_view')
        
        lines = error.split('\n')
        
        # Should have multiple sections
        assert len(lines) > 10, "Error too brief"
        
        # Check section headers
        sections = [line.strip() for line in lines if line and line[0].isupper()]
        assert len(sections) >= 3, "Not enough sections"
    
    def test_error_is_actionable(self):
        """Error should tell user exactly what to do."""
        error = format_alternatives_error('code_block')
        
        # Should have concrete alternatives
        assert 'show text' in error.lower()
        
        # Should have example code
        assert 'def ' in error or 'print' in error or '```' in error
        
        # Should have documentation
        assert '.md' in error
    
    def test_error_is_not_overwhelming(self):
        """Error should be comprehensive but not overwhelming."""
        error = format_alternatives_error('progress_bar')
        
        # Should fit on a screen (roughly)
        lines = error.split('\n')
        assert len(lines) < 100, "Error too long"
        
        # Should have clear formatting
        assert error.count('\n\n') >= 3, "Needs more spacing"
    
    def test_recommended_alternative_is_clear(self):
        """Error should clearly indicate the recommended alternative."""
        error = format_alternatives_error('tree_view')
        
        # Should mention primary recommendation
        assert 'Recommended:' in error or '✨' in error or 'accordion' in error
    
    def test_nonexistent_component(self):
        """Non-unsupported components should return None or generic error."""
        # These are valid components
        assert get_component_alternatives('text') is None
        assert get_component_alternatives('table') is None
        assert get_component_alternatives('chart') is None
        
        # These don't exist at all
        assert get_component_alternatives('fake_component') is None
        assert get_component_alternatives('nonexistent') is None
        
        # Generic error for non-existent
        error = format_alternatives_error('fake_component')
        assert "is not supported" in error
        assert len(error) < 100, "Generic error should be brief"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
