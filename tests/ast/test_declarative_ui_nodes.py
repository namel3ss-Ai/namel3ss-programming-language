"""
Tests for declarative UI AST nodes (ShowCard, ShowList, etc.).

These tests validate that the new declarative UI node types are properly
defined and can be instantiated with correct types.
"""

import pytest
from namel3ss.ast.pages import (
    BadgeConfig,
    CardFooter,
    CardHeader,
    CardItemConfig,
    CardSection,
    ConditionalAction,
    EmptyStateConfig,
    FieldValueConfig,
    InfoGridItem,
    ShowCard,
    ShowList,
)


def test_empty_state_config_creation():
    """EmptyStateConfig should accept all optional fields."""
    empty_state = EmptyStateConfig(
        icon="calendar",
        icon_size="large",
        title="No appointments",
        message="Check back later",
        action_label="Schedule Now",
        action_link="/schedule",
    )
    
    assert empty_state.icon == "calendar"
    assert empty_state.icon_size == "large"
    assert empty_state.title == "No appointments"
    assert empty_state.message == "Check back later"


def test_badge_config_with_condition():
    """BadgeConfig should support conditional rendering."""
    badge = BadgeConfig(
        field="status",
        style="status_badge",
        transform="humanize",
        condition="status != null",
    )
    
    assert badge.field == "status"
    assert badge.condition == "status != null"
    assert badge.transform == "humanize"


def test_field_value_config():
    """FieldValueConfig should handle both fields and static text."""
    # Field-based value
    field_value = FieldValueConfig(
        field="date",
        format="MMMM DD, YYYY",
        style="date_value",
    )
    assert field_value.field == "date"
    assert field_value.format == "MMMM DD, YYYY"
    
    # Static text value
    text_value = FieldValueConfig(
        text="Dr. {{ provider }}",
    )
    assert text_value.text == "Dr. {{ provider }}"


def test_info_grid_item():
    """InfoGridItem should handle icon, label, and multiple values."""
    grid_item = InfoGridItem(
        icon="calendar",
        label="Date & Time",
        values=[
            FieldValueConfig(field="date", format="MMMM DD, YYYY"),
            FieldValueConfig(field="time"),
        ],
    )
    
    assert grid_item.icon == "calendar"
    assert grid_item.label == "Date & Time"
    assert len(grid_item.values) == 2
    assert grid_item.values[0].field == "date"


def test_card_section_info_grid():
    """CardSection of type info_grid should have columns and items."""
    section = CardSection(
        type="info_grid",
        columns=2,
        items=[
            InfoGridItem(icon="calendar", label="Date", values=[FieldValueConfig(field="date")]),
            InfoGridItem(icon="user", label="Provider", values=[FieldValueConfig(field="provider")]),
        ],
    )
    
    assert section.type == "info_grid"
    assert section.columns == 2
    assert len(section.items) == 2


def test_card_section_with_condition():
    """CardSection should support conditional rendering."""
    section = CardSection(
        type="text_section",
        condition="reason != null",
        content={"label": "Reason:", "text": "{{ reason }}"},
    )
    
    assert section.type == "text_section"
    assert section.condition == "reason != null"
    assert section.content["label"] == "Reason:"


def test_conditional_action():
    """ConditionalAction should support conditions and multiple trigger types."""
    # Action with condition
    action = ConditionalAction(
        label="Cancel",
        icon="x",
        style="danger",
        action="cancel_appointment",
        params="{{ id }}",
        condition="status == 'pending'",
    )
    
    assert action.label == "Cancel"
    assert action.condition == "status == 'pending'"
    assert action.action == "cancel_appointment"
    
    # Link-based action
    link_action = ConditionalAction(
        label="View Details",
        link="/appointments/{{ id }}",
    )
    assert link_action.link == "/appointments/{{ id }}"


def test_card_header_with_badges():
    """CardHeader should support title, subtitle, and multiple badges."""
    header = CardHeader(
        title="{{ type }}",
        subtitle="{{ provider }}",
        badges=[
            BadgeConfig(field="status", transform="humanize"),
            BadgeConfig(field="priority", style="priority_badge"),
        ],
    )
    
    assert header.title == "{{ type }}"
    assert len(header.badges) == 2
    assert header.badges[0].field == "status"


def test_card_footer_with_condition():
    """CardFooter should support conditional display."""
    footer = CardFooter(
        condition="confirmation_number != null",
        text="Confirmation #: {{ confirmation_number }}",
        style="confirmation",
    )
    
    assert footer.condition == "confirmation_number != null"
    assert footer.text == "Confirmation #: {{ confirmation_number }}"


def test_card_item_config_complete():
    """CardItemConfig should support full card structure."""
    item_config = CardItemConfig(
        type="card",
        style="appointment_detail",
        state_class={"urgent": "priority == 'high'"},
        header=CardHeader(
            title="{{ type }}",
            badges=[BadgeConfig(field="status")],
        ),
        sections=[
            CardSection(
                type="info_grid",
                columns=2,
                items=[InfoGridItem(label="Date", values=[FieldValueConfig(field="date")])],
            ),
        ],
        actions=[
            ConditionalAction(label="Edit", action="edit", condition="editable == true"),
        ],
        footer=CardFooter(text="Footer text"),
    )
    
    assert item_config.type == "card"
    assert item_config.style == "appointment_detail"
    assert "urgent" in item_config.state_class
    assert isinstance(item_config.header, CardHeader)
    assert len(item_config.sections) == 1
    assert len(item_config.actions) == 1
    assert isinstance(item_config.footer, CardFooter)


def test_show_card_statement():
    """ShowCard should be a valid PageStatement with all configuration."""
    card = ShowCard(
        title="My Appointments",
        source_type="dataset",
        source="appointments",
        empty_state=EmptyStateConfig(
            icon="calendar",
            title="No appointments",
        ),
        item_config=CardItemConfig(
            type="card",
            sections=[
                CardSection(type="info_grid", columns=2, items=[]),
            ],
        ),
        group_by="date",
        filter_by="status == 'active'",
    )
    
    assert card.title == "My Appointments"
    assert card.source_type == "dataset"
    assert card.source == "appointments"
    assert card.empty_state.icon == "calendar"
    assert card.item_config.type == "card"
    assert card.group_by == "date"


def test_show_list_statement():
    """ShowList should support list-specific configuration."""
    list_widget = ShowList(
        title="Messages",
        source_type="dataset",
        source="messages",
        list_type="conversation",
        enable_search=True,
        search_placeholder="Search messages...",
        filters=[
            {"label": "All", "value": "all"},
            {"label": "Unread", "value": "unread"},
        ],
        columns=2,
    )
    
    assert list_widget.title == "Messages"
    assert list_widget.list_type == "conversation"
    assert list_widget.enable_search is True
    assert len(list_widget.filters) == 2
    assert list_widget.columns == 2


def test_show_card_minimal():
    """ShowCard should work with minimal configuration."""
    card = ShowCard(
        title="Simple Card",
        source_type="dataset",
        source="items",
    )
    
    assert card.title == "Simple Card"
    assert card.source == "items"
    assert card.empty_state is None
    assert card.item_config is None


def test_nested_dict_content():
    """Sections should support nested dict content for flexibility."""
    section = CardSection(
        type="text_section",
        content={
            "label": "Description:",
            "text": "{{ description }}",
            "style": "description_text",
        },
    )
    
    assert section.content["label"] == "Description:"
    assert section.content["text"] == "{{ description }}"


def test_badge_with_transform_dict():
    """Badge transforms can be strings or dicts for complex formatting."""
    # String transform
    badge1 = BadgeConfig(field="status", transform="humanize")
    assert badge1.transform == "humanize"
    
    # Dict transform
    badge2 = BadgeConfig(
        field="date",
        transform={"format": "MMMM DD, YYYY", "locale": "en-US"},
    )
    assert isinstance(badge2.transform, dict)
    assert badge2.transform["format"] == "MMMM DD, YYYY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
