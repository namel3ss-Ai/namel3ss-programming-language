"""
Test advanced declarative UI features:
- type: card
- type: info_grid  
- empty_state: { icon, title }
- sections: [{ type: info_grid, ... }]
- header.badges
- item.actions with conditional display (conditions bound to real data/state)
- Semantic component types: message_bubble, article_card, etc.
"""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast.pages import (
    ShowCard, 
    ShowList, 
    EmptyStateConfig, 
    CardItemConfig,
    CardHeader,
    CardSection,
    ConditionalAction,
    BadgeConfig,
    InfoGridItem,
)


def test_card_type_property():
    """Test parsing card with explicit type: card."""
    source = '''
app "Test App"

dataset "appointments" from "db://appointments"

page "Appointments" at "/appointments":
  show card "Appointments" from dataset appointments:
    item:
      type: card
      style: elevated
      
      header:
        title: patient_name
        subtitle: appointment_type
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.type == "card"
    assert statement.item_config.style == "elevated"
    print("✓ Card type property parsed correctly")


def test_empty_state_with_icon_and_title():
    """Test parsing empty_state with icon and title."""
    source = '''
app "Test App"

dataset "tasks" from "api://tasks"

page "Tasks" at "/tasks":
  show card "Tasks" from dataset tasks:
    empty_state:
      icon: inbox
      title: "No tasks yet"
      message: "Create your first task to get started"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.empty_state is not None
    assert isinstance(statement.empty_state, EmptyStateConfig)
    assert statement.empty_state.icon == "inbox"
    assert statement.empty_state.title == "No tasks yet"
    assert statement.empty_state.message == "Create your first task to get started"
    print("✓ Empty state with icon and title parsed correctly")


def test_sections_with_info_grid_type():
    """Test parsing sections with type: info_grid."""
    source = '''
app "Test App"

dataset "patients" from "db://patients"

page "Patients" at "/patients":
  show card "Patient Records" from dataset patients:
    item:
      type: card
      
      sections:
        - type: info_grid
          columns: 2
          title: "Patient Details"
          items:
            - icon: user
              label: "Name"
              values:
                - field: full_name
            - icon: calendar
              label: "DOB"
              values:
                - field: date_of_birth
                  format: "MMM DD, YYYY"
        
        - type: info_grid
          columns: 3
          title: "Contact Information"
          items:
            - icon: phone
              label: "Phone"
              values:
                - field: phone_number
            - icon: mail
              label: "Email"
              values:
                - field: email
            - icon: home
              label: "Address"
              values:
                - field: address
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.sections is not None
    assert len(statement.item_config.sections) == 2
    
    # Check first section
    section1 = statement.item_config.sections[0]
    assert section1.type == "info_grid"
    assert section1.columns == 2
    assert section1.title == "Patient Details"
    assert len(section1.items) == 2
    assert section1.items[0].icon == "user"
    assert section1.items[0].label == "Name"
    
    # Check second section
    section2 = statement.item_config.sections[1]
    assert section2.type == "info_grid"
    assert section2.columns == 3
    assert section2.title == "Contact Information"
    assert len(section2.items) == 3
    
    print("✓ Sections with info_grid type parsed correctly")


def test_header_badges():
    """Test parsing header.badges with multiple badges."""
    source = '''
app "Test App"

dataset "orders" from "db://orders"

page "Orders" at "/orders":
  show card "Orders" from dataset orders:
    item:
      type: card
      
      header:
        title: order_id
        subtitle: customer_name
        badges:
          - field: status
            style: status-badge
            transform: uppercase
          - field: priority
            style: priority-badge
            icon: flag
          - text: "Urgent"
            style: urgent-badge
            condition: "priority == 'high'"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.header is not None
    assert isinstance(statement.item_config.header, CardHeader)
    assert statement.item_config.header.badges is not None
    assert len(statement.item_config.header.badges) == 3
    
    # Check first badge
    badge1 = statement.item_config.header.badges[0]
    assert badge1.field == "status"
    assert badge1.style == "status-badge"
    assert badge1.transform == "uppercase"
    
    # Check second badge
    badge2 = statement.item_config.header.badges[1]
    assert badge2.field == "priority"
    assert badge2.style == "priority-badge"
    assert badge2.icon == "flag"
    
    # Check third badge (conditional)
    badge3 = statement.item_config.header.badges[2]
    assert badge3.text == "Urgent"
    assert badge3.style == "urgent-badge"
    assert badge3.condition == "priority == 'high'"
    
    print("✓ Header badges parsed correctly")


def test_item_actions_with_conditionals():
    """Test parsing item.actions with conditional display bound to real data."""
    source = '''
app "Test App"

dataset "tickets" from "api://tickets"

page "Tickets" at "/tickets":
  show card "Support Tickets" from dataset tickets:
    item:
      type: card
      
      header:
        title: ticket_id
        subtitle: subject
      
      actions:
        - label: "Assign"
          icon: user-plus
          action: assign_ticket
          style: primary
          condition: "status == 'open' and assigned_to == null"
        
        - label: "Close"
          icon: check-circle
          action: close_ticket
          style: success
          condition: "status == 'open' or status == 'in_progress'"
        
        - label: "Reopen"
          icon: refresh
          action: reopen_ticket
          style: secondary
          condition: "status == 'closed'"
        
        - label: "Escalate"
          icon: arrow-up
          action: escalate_ticket
          style: danger
          condition: "priority == 'low' and days_open > 7"
        
        - label: "View Details"
          icon: eye
          link: "/tickets/{{ id }}"
          style: secondary
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.actions is not None
    assert len(statement.item_config.actions) == 5
    
    # Check assign action (conditional on status and assigned_to)
    action1 = statement.item_config.actions[0]
    assert action1.label == "Assign"
    assert action1.icon == "user-plus"
    assert action1.action == "assign_ticket"
    assert action1.style == "primary"
    assert action1.condition == "status == 'open' and assigned_to == null"
    
    # Check close action (conditional on status)
    action2 = statement.item_config.actions[1]
    assert action2.label == "Close"
    assert action2.condition == "status == 'open' or status == 'in_progress'"
    
    # Check reopen action (conditional on closed status)
    action3 = statement.item_config.actions[2]
    assert action3.label == "Reopen"
    assert action3.condition == "status == 'closed'"
    
    # Check escalate action (conditional on priority and days)
    action4 = statement.item_config.actions[3]
    assert action4.label == "Escalate"
    assert action4.condition == "priority == 'low' and days_open > 7"
    
    # Check view action (no condition - always visible)
    action5 = statement.item_config.actions[4]
    assert action5.label == "View Details"
    assert action5.link == "/tickets/{{ id }}"
    assert action5.condition is None
    
    print("✓ Item actions with conditionals parsed correctly")


def test_semantic_component_message_bubble():
    """Test parsing semantic component type: message_bubble."""
    source = '''
app "Test App"

dataset "messages" from "api://messages"

page "Chat" at "/chat":
  show card "Messages" from dataset messages:
    item:
      type: message_bubble
      role_class: "{{ sender_role }}"
      
      header:
        title: sender_name
        subtitle: timestamp
        avatar:
          field: sender_avatar
          fallback: "{{ sender_name[0] }}"
      
      sections:
        - type: text_section
          content:
            text: message_content
      
      actions:
        - label: "Reply"
          icon: reply
          action: reply_message
        - label: "React"
          icon: heart
          action: react_message
          condition: "can_react == true"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.type == "message_bubble"
    assert statement.item_config.role_class == "{{ sender_role }}"
    assert statement.item_config.header is not None
    assert statement.item_config.header.avatar is not None
    print("✓ Semantic component type message_bubble parsed correctly")


def test_semantic_component_article_card():
    """Test parsing semantic component type: article_card."""
    source = '''
app "Test App"

dataset "articles" from "api://articles"

page "Blog" at "/blog":
  show card "Articles" from dataset articles:
    item:
      type: article_card
      style: featured
      
      header:
        title: article_title
        subtitle: author_name
        badges:
          - field: category
            style: category-badge
          - field: read_time
            icon: clock
            transform: 
              format: "{{ value }} min read"
      
      sections:
        - type: text_section
          content:
            text: excerpt
            style: preview
        
        - type: info_grid
          columns: 3
          items:
            - icon: calendar
              label: "Published"
              values:
                - field: published_date
                  format: "MMM DD, YYYY"
            - icon: eye
              label: "Views"
              values:
                - field: view_count
            - icon: heart
              label: "Likes"
              values:
                - field: like_count
      
      actions:
        - label: "Read More"
          icon: arrow-right
          link: "/articles/{{ slug }}"
          style: primary
        - label: "Share"
          icon: share
          action: share_article
          style: secondary
        - label: "Bookmark"
          icon: bookmark
          action: bookmark_article
          condition: "is_bookmarked == false"
        - label: "Remove Bookmark"
          icon: bookmark-fill
          action: remove_bookmark
          condition: "is_bookmarked == true"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    assert statement.item_config.type == "article_card"
    assert statement.item_config.style == "featured"
    
    # Check sections
    assert len(statement.item_config.sections) == 2
    assert statement.item_config.sections[0].type == "text_section"
    assert statement.item_config.sections[1].type == "info_grid"
    
    # Check conditional bookmark actions
    assert len(statement.item_config.actions) == 4
    bookmark_action = statement.item_config.actions[2]
    remove_bookmark_action = statement.item_config.actions[3]
    assert bookmark_action.condition == "is_bookmarked == false"
    assert remove_bookmark_action.condition == "is_bookmarked == true"
    
    print("✓ Semantic component type article_card parsed correctly")


def test_complex_conditions_with_data_state():
    """Test complex conditional expressions bound to data and state."""
    source = '''
app "Test App"

dataset "tasks" from "db://tasks"

page "Tasks" at "/tasks":
  show card "Tasks" from dataset tasks:
    item:
      type: card
      state_class:
        overdue: "due_date < today() and status != 'completed'"
        urgent: "priority == 'high' and status == 'pending'"
        completed: "status == 'completed'"
      
      header:
        title: task_name
        badges:
          - text: "OVERDUE"
            style: danger
            condition: "due_date < today() and status != 'completed'"
          - field: priority
            style: "priority-{{ priority }}"
            condition: "priority == 'high' or priority == 'critical'"
          - text: "✓ Done"
            style: success
            condition: "status == 'completed'"
      
      sections:
        - type: info_grid
          columns: 2
          condition: "status != 'completed'"
          items:
            - label: "Due Date"
              values:
                - field: due_date
                  format: "MMM DD, YYYY"
            - label: "Assigned To"
              values:
                - field: assigned_to
      
      actions:
        - label: "Start"
          icon: play
          action: start_task
          style: primary
          condition: "status == 'pending' and assigned_to == current_user.id"
        
        - label: "Complete"
          icon: check
          action: complete_task
          style: success
          condition: "status == 'in_progress' and assigned_to == current_user.id"
        
        - label: "Reassign"
          icon: users
          action: reassign_task
          condition: "status != 'completed' and (role == 'manager' or role == 'admin')"
        
        - label: "Delete"
          icon: trash
          action: delete_task
          style: danger
          condition: "status == 'completed' or (created_by == current_user.id and status == 'pending')"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.item_config is not None
    
    # Check state_class with complex conditions
    assert statement.item_config.state_class is not None
    assert "overdue" in statement.item_config.state_class
    assert "due_date < today() and status != 'completed'" in statement.item_config.state_class["overdue"]
    
    # Check badges with conditions
    assert len(statement.item_config.header.badges) == 3
    assert statement.item_config.header.badges[0].condition == "due_date < today() and status != 'completed'"
    assert statement.item_config.header.badges[1].condition == "priority == 'high' or priority == 'critical'"
    
    # Check section with condition
    assert statement.item_config.sections[0].condition == "status != 'completed'"
    
    # Check actions with complex conditions
    assert len(statement.item_config.actions) == 4
    assert "current_user.id" in statement.item_config.actions[0].condition
    assert "role == 'manager' or role == 'admin'" in statement.item_config.actions[2].condition
    assert "created_by == current_user.id" in statement.item_config.actions[3].condition
    
    print("✓ Complex conditions with data state parsed correctly")


def test_all_features_combined():
    """Test all advanced features combined in one comprehensive example."""
    source = '''
app "Healthcare App"

dataset "appointments" from "db://appointments"

page "Appointments" at "/appointments":
  show card "Patient Appointments" from dataset appointments:
    empty_state:
      icon: calendar-x
      icon_size: large
      title: "No appointments scheduled"
      message: "Schedule your first appointment to get started"
      action_label: "Schedule Appointment"
      action_link: "/appointments/new"
    
    filter_by: "status != 'cancelled'"
    sort_by: "appointment_date asc"
    group_by: "appointment_date"
    
    item:
      type: card
      style: elevated
      state_class:
        upcoming: "status == 'scheduled' and appointment_date >= today()"
        past: "status == 'completed'"
        missed: "status == 'scheduled' and appointment_date < today()"
      
      header:
        title: patient_name
        subtitle: "{{ appointment_type }} with Dr. {{ provider }}"
        badges:
          - field: status
            style: "status-{{ status }}"
            transform: humanize
            icon: circle
          - text: "NEW"
            style: new-badge
            condition: "is_first_visit == true"
          - text: "URGENT"
            style: urgent-badge
            icon: alert
            condition: "priority == 'urgent'"
        avatar:
          field: patient_avatar
          fallback: "{{ patient_name[0] }}"
          size: md
      
      sections:
        - type: info_grid
          title: "Appointment Details"
          columns: 2
          items:
            - icon: calendar
              label: "Date & Time"
              values:
                - field: appointment_date
                  format: "MMMM DD, YYYY"
                - field: appointment_time
                  format: "h:mm A"
            - icon: user-md
              label: "Provider"
              values:
                - text: "Dr. {{ provider }}"
            - icon: building
              label: "Location"
              values:
                - field: clinic_name
                - field: room_number
                  text: "Room {{ room_number }}"
            - icon: clock
              label: "Duration"
              values:
                - field: duration
                  format: "{{ value }} minutes"
        
        - type: info_grid
          title: "Patient Information"
          columns: 3
          condition: "role == 'provider' or role == 'staff'"
          items:
            - icon: phone
              label: "Phone"
              values:
                - field: patient_phone
            - icon: mail
              label: "Email"
              values:
                - field: patient_email
            - icon: calendar-check
              label: "DOB"
              values:
                - field: patient_dob
                  format: "MM/DD/YYYY"
        
        - type: text_section
          title: "Notes"
          condition: "notes != null and notes != ''"
          content:
            text: notes
            style: notes-section
      
      actions:
        - label: "Check In"
          icon: check-circle
          action: check_in_patient
          style: primary
          condition: "status == 'scheduled' and appointment_date == today() and check_in_time == null"
        
        - label: "Start Appointment"
          icon: play
          action: start_appointment
          style: success
          condition: "status == 'checked_in' and (role == 'provider' or role == 'staff')"
        
        - label: "Complete"
          icon: check
          action: complete_appointment
          style: success
          condition: "status == 'in_progress' and role == 'provider'"
        
        - label: "Reschedule"
          icon: calendar
          action: reschedule_appointment
          style: secondary
          condition: "status == 'scheduled' and appointment_date >= today()"
        
        - label: "Cancel"
          icon: x-circle
          action: cancel_appointment
          style: danger
          condition: "status == 'scheduled' and (role == 'patient' or role == 'staff' or role == 'provider')"
        
        - label: "View Details"
          icon: eye
          link: "/appointments/{{ id }}"
          style: secondary
        
        - label: "Send Reminder"
          icon: bell
          action: send_reminder
          condition: "status == 'scheduled' and reminder_sent == false and (role == 'staff' or role == 'admin')"
      
      footer:
        condition: "status == 'completed'"
        text: "Completed on {{ completed_date }}"
        style: completed-footer
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    statement = page.body[0]
    
    # Verify all components
    assert statement.__class__.__name__ == "ShowCard"
    assert statement.title == "Patient Appointments"
    
    # Verify empty_state
    assert statement.empty_state is not None
    assert statement.empty_state.icon == "calendar-x"
    assert statement.empty_state.icon_size == "large"
    assert statement.empty_state.action_label is not None
    
    # Verify item config
    assert statement.item_config is not None
    assert statement.item_config.type == "card"
    assert statement.item_config.style == "elevated"
    
    # Verify state_class
    assert statement.item_config.state_class is not None
    assert "upcoming" in statement.item_config.state_class
    assert "past" in statement.item_config.state_class
    
    # Verify header with badges
    assert statement.item_config.header is not None
    assert len(statement.item_config.header.badges) == 3
    assert statement.item_config.header.avatar is not None
    
    # Verify sections
    assert len(statement.item_config.sections) == 3
    assert statement.item_config.sections[0].type == "info_grid"
    assert statement.item_config.sections[0].columns == 2
    assert statement.item_config.sections[1].condition is not None
    assert statement.item_config.sections[2].type == "text_section"
    
    # Verify actions
    assert len(statement.item_config.actions) == 7
    check_in = statement.item_config.actions[0]
    assert check_in.label == "Check In"
    assert "appointment_date == today()" in check_in.condition
    
    # Verify footer
    assert statement.item_config.footer is not None
    assert statement.item_config.footer.condition == "status == 'completed'"
    
    print("✓ All advanced features combined parsed correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
