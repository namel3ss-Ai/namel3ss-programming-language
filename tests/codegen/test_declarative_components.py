"""
Test frontend codegen for declarative card components with semantic HTML and accessibility.

Tests validate:
- Semantic HTML elements (article, section, header, nav, footer)
- Accessibility attributes (aria-label, aria-labelledby, role)
- Responsive CSS Grid for info_grid
- Semantic component type differentiation (card, message_bubble, article_card)
- Design system token usage (--spacing-*, --font-size-*, --border-color, etc.)
"""

import pytest
from pathlib import Path
import tempfile


def get_card_widget_code() -> str:
    """Generate and read CardWidget.tsx file."""
    from namel3ss.codegen.frontend.react.declarative_components import write_card_widget
    
    with tempfile.TemporaryDirectory() as tmpdir:
        components_dir = Path(tmpdir) / "components"
        components_dir.mkdir()
        
        write_card_widget(components_dir)
        
        card_widget_path = components_dir / "CardWidget.tsx"
        return card_widget_path.read_text(encoding='utf-8')


# =============================================================================
# SEMANTIC HTML TESTS
# =============================================================================


def test_card_uses_article_element():
    """Test that cards use <article> element instead of <div>."""
    code = get_card_widget_code()
    
    # Should use <article> for card items
    assert '<article' in code
    assert 'n3-card-item' in code


def test_card_header_uses_header_element():
    """Test that card headers use <header> element."""
    code = get_card_widget_code()
    
    # Should use <header> for card header
    assert '<header' in code
    assert 'n3-card-header' in code


def test_card_sections_use_section_element():
    """Test that card sections use <section> element."""
    code = get_card_widget_code()
    
    # Info grid should use <section>
    assert '<section' in code
    assert 'info-grid' in code


def test_card_actions_use_nav_element():
    """Test that card actions use <nav> element."""
    code = get_card_widget_code()
    
    # Actions should use <nav>
    assert '<nav' in code
    assert 'n3-card-actions' in code
    assert 'aria-label="Card actions"' in code


def test_card_footer_uses_footer_element():
    """Test that card footer uses <footer> element."""
    code = get_card_widget_code()
    
    # Footer should use <footer>
    assert '<footer' in code
    assert 'n3-card-footer' in code


# =============================================================================
# ACCESSIBILITY TESTS
# =============================================================================


def test_card_has_aria_labelledby():
    """Test that cards have aria-labelledby pointing to header."""
    code = get_card_widget_code()
    
    # Card should have aria-labelledby
    assert 'aria-labelledby={headerId}' in code
    
    # Header should have id
    assert 'id={headerId}' in code


def test_widget_section_has_aria_labelledby():
    """Test that widget section has aria-labelledby pointing to title."""
    code = get_card_widget_code()
    
    # Widget section should have aria-labelledby
    assert 'aria-labelledby={titleId}' in code
    
    # Title should have id
    assert 'id={titleId}' in code


def test_info_grid_has_aria_labelledby_support():
    """Test that info_grid supports aria-labelledby."""
    code = get_card_widget_code()
    
    # Info grid should support aria-labelledby when title is present
    assert 'aria-labelledby={titleId}' in code


def test_empty_state_has_role_status():
    """Test that empty state has role="status" for screen readers."""
    code = get_card_widget_code()
    
    # Empty state should have role="status"
    assert 'role="status"' in code
    assert 'aria-label="No items to display"' in code


def test_action_buttons_have_aria_label():
    """Test that action buttons have aria-label."""
    code = get_card_widget_code()
    
    # Buttons should have aria-label
    assert 'aria-label={action.label}' in code


def test_badges_have_role_list():
    """Test that badge container has role="list"."""
    code = get_card_widget_code()
    
    # Badges container should have role="list"
    assert 'role="list"' in code
    assert 'aria-label="Status badges"' in code
    
    # Individual badges should have role="listitem"
    assert 'role="listitem"' in code


def test_decorative_icons_have_aria_hidden():
    """Test that decorative icons have aria-hidden="true"."""
    code = get_card_widget_code()
    
    # Decorative elements should have aria-hidden
    assert 'aria-hidden="true"' in code or 'aria-hidden={true}' in code


# =============================================================================
# RESPONSIVE CSS GRID TESTS
# =============================================================================


def test_info_grid_uses_css_grid_repeat_auto_fit():
    """Test that info_grid uses CSS Grid with repeat(auto-fit, minmax())."""
    code = get_card_widget_code()
    
    # Should use CSS Grid with repeat(auto-fit, minmax())
    assert 'display: \'grid\'' in code or 'display: "grid"' in code
    assert 'gridTemplateColumns' in code
    assert 'repeat(auto-fit, minmax(' in code


def test_info_grid_respects_column_count():
    """Test that info_grid respects column configuration."""
    code = get_card_widget_code()
    
    # Should reference section.columns
    assert 'section.columns' in code


def test_info_grid_uses_design_tokens():
    """Test that info_grid uses design system spacing tokens."""
    code = get_card_widget_code()
    
    # Should use CSS variables for spacing
    assert 'var(--spacing-' in code
    
    # Should use gap property
    assert 'gap:' in code


# =============================================================================
# SEMANTIC COMPONENT TYPE TESTS
# =============================================================================


def test_message_bubble_type_uses_aria_role():
    """Test that message_bubble type uses proper ARIA role."""
    code = get_card_widget_code()
    
    # Should detect message_bubble type
    assert 'message_bubble' in code
    
    # Should set aria role
    assert 'role={ariaRole}' in code
    assert 'isMessageBubble' in code


def test_article_card_type_identified():
    """Test that article_card type is properly identified."""
    code = get_card_widget_code()
    
    # Should detect article_card type
    assert 'article_card' in code
    assert 'isArticle' in code


def test_card_type_classes_include_semantic_type():
    """Test that card className includes semantic type (n3-card-card, n3-card-message_bubble, etc.)."""
    code = get_card_widget_code()
    
    # Should include semantic type in className
    assert 'n3-card-${cardType}' in code


# =============================================================================
# DESIGN SYSTEM TOKEN TESTS
# =============================================================================


def test_card_uses_design_tokens_for_spacing():
    """Test that cards use design system tokens for spacing."""
    code = get_card_widget_code()
    
    # Should use var(--spacing-*) tokens
    assert 'var(--spacing-lg' in code or 'var(--spacing-md' in code
    assert 'var(--spacing-sm' in code or 'var(--spacing-xs' in code


def test_card_uses_design_tokens_for_colors():
    """Test that cards use design system tokens for colors."""
    code = get_card_widget_code()
    
    # Should use color tokens
    assert 'var(--border-color' in code
    assert 'var(--text-primary' in code or 'var(--text-secondary' in code
    assert 'var(--surface' in code


def test_card_uses_design_tokens_for_typography():
    """Test that cards use design system tokens for typography."""
    code = get_card_widget_code()
    
    # Should use font-size tokens
    assert 'var(--font-size-' in code


def test_card_uses_design_tokens_for_radius():
    """Test that cards use design system tokens for border radius."""
    code = get_card_widget_code()
    
    # Should use radius tokens
    assert 'var(--radius-' in code


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_all_semantic_elements_present():
    """Test that all semantic elements are present in generated code."""
    code = get_card_widget_code()
    
    # Should have all semantic elements
    assert '<article' in code
    assert '<header' in code
    assert '<section' in code
    assert '<nav' in code
    assert '<footer' in code


def test_comprehensive_accessibility():
    """Test that comprehensive accessibility features are present."""
    code = get_card_widget_code()
    
    # Should have accessibility attributes
    assert 'aria-labelledby' in code
    assert 'aria-label' in code
    assert 'role=' in code
    assert 'aria-hidden' in code


def test_design_system_integration():
    """Test that design system tokens are used throughout."""
    code = get_card_widget_code()
    
    # Should use design tokens
    assert 'var(--spacing-' in code
    assert 'var(--border-color' in code
    assert 'var(--font-size-' in code
    assert 'var(--radius-' in code
    assert 'var(--text-' in code or 'var(--surface' in code


def test_responsive_grid_implementation():
    """Test that responsive grid is properly implemented."""
    code = get_card_widget_code()
    
    # Should have responsive grid
    assert 'repeat(auto-fit, minmax(' in code
    assert 'gridTemplateColumns' in code


def test_no_demo_data_or_placeholders():
    """Test that generated code contains no demo data, placeholders, or TODOs."""
    code = get_card_widget_code()
    
    # Should NOT contain demo data or placeholders
    assert 'TODO' not in code
    assert 'FIXME' not in code
    # Note: "placeholder" might appear in variable names like "searchPlaceholder", that's OK
    # But shouldn't appear as demo content
    assert 'demo data' not in code.lower()
    assert 'lorem ipsum' not in code.lower()


def test_proper_heading_hierarchy():
    """Test that proper heading hierarchy is used (h2 for widget, h3 for card)."""
    code = get_card_widget_code()
    
    # Widget title should use h2
    assert '<h2' in code
    assert 'n3-widget-title' in code
    
    # Card title should use h3
    assert '<h3' in code
    assert 'n3-card-title' in code


def test_proper_dt_dd_in_info_grid():
    """Test that info grid uses proper <dt> and <dd> elements for label/value pairs."""
    code = get_card_widget_code()
    
    # Should use definition list elements
    assert '<dt' in code
    assert 'info-grid-label' in code
    assert '<dd' in code
    assert 'info-grid-value' in code
