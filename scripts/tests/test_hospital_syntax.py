#!/usr/bin/env python3
"""
Test parsing the hospital-ai patient appointments syntax.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from namel3ss.parser import Parser
from namel3ss.ast.pages import ShowCard

def test_hospital_appointments_syntax():
    """Test the exact syntax from hospital-ai ui_patient.ai."""
    print("Testing hospital-ai appointments page syntax...")
    
    # This is the exact syntax from the selected code
    source = '''
dataset appointments:
  fields:
    - id: int
    - type: text
    - status: text
    - date: date
    - time: text
    - provider: text
    - specialty: text
    - location: text
    - duration: int
    - reason: text
    - instructions: text
    - confirmation_number: text
    - days_until: int

page patient_appointments:
  path: "/patient/appointments"
  title: "My Appointments"
  
  show card "My Appointments" from dataset appointments:
    group_by: "date"
    
    empty_state:
      icon: calendar
      icon_size: large
      title: "No appointments scheduled"
      message: "Your care team will schedule appointments as needed."
    
    item:
      type: card
      style: appointment_detail
      
      header:
        badges:
          - field: type
            style: type_badge
            transform: humanize
          - field: status
            style: status_badge
            transform: humanize
      
      sections:
        - type: info_grid
          columns: 2
          items:
            - icon: calendar
              label: "Date & Time"
              values:
                - field: date
                  format: "MMMM DD, YYYY"
                - field: time
            
            - icon: user
              label: "Provider"
              values:
                - text: "Dr. {{ provider }}"
                - field: specialty
                  style: sub_value
            
            - icon: map-pin
              label: "Location"
              values:
                - field: location
            
            - icon: clock
              label: "Duration"
              values:
                - text: "{{ duration }} minutes"
        
        - type: text_section
          condition: "reason != null"
          style: reason
          content:
            label: "Reason for visit:"
            text: "{{ reason }}"
        
        - type: text_section
          condition: "instructions != null"
          style: instructions
          icon: info-circle
          content:
            label: "Instructions:"
            text: "{{ instructions }}"
      
      actions:
        - label: "Request Reschedule"
          icon: calendar
          style: secondary
          action: request_reschedule
          params: "{{ id }}"
          condition: "status == 'confirmed' && days_until > 2"
        
        - label: "Print Confirmation"
          icon: printer
          style: secondary
          action: print_confirmation
          params: "{{ id }}"
          condition: "confirmation_number != null"
        
        - label: "Add to Calendar"
          icon: download
          style: secondary
          action: add_to_calendar
          params: "{{ id }}"
          condition: "days_until <= 7 && days_until >= 0"
      
      footer:
        condition: "confirmation_number != null"
        text: "Confirmation #: {{ confirmation_number }}"
        style: confirmation
'''
    
    try:
        parser = Parser()
        app = parser.parse(source)
        
        assert len(app.pages) == 1, f"Expected 1 page, got {len(app.pages)}"
        page = app.pages[0]
        
        print(f"   Page name: {page.name}")
        print(f"   Page path: {page.path}")
        print(f"   Page title: {page.title}")
        
        assert len(page.statements) == 1, f"Expected 1 statement, got {len(page.statements)}"
        statement = page.statements[0]
        
        assert isinstance(statement, ShowCard), f"Expected ShowCard, got {type(statement)}"
        print(f"   Card title: {statement.title}")
        print(f"   Card source: {statement.source_type} {statement.source}")
        print(f"   Group by: {statement.group_by}")
        
        # Check empty state
        assert statement.empty_state is not None, "empty_state is None"
        print(f"   Empty state icon: {statement.empty_state.icon}")
        print(f"   Empty state icon_size: {statement.empty_state.icon_size}")
        print(f"   Empty state title: {statement.empty_state.title}")
        assert statement.empty_state.icon == "calendar"
        assert statement.empty_state.icon_size == "large"
        
        # Check item config
        assert statement.item_config is not None, "item_config is None"
        assert statement.item_config.type == "card"
        assert statement.item_config.style == "appointment_detail"
        print(f"   Item type: {statement.item_config.type}")
        print(f"   Item style: {statement.item_config.style}")
        
        # Check header badges
        assert statement.item_config.header is not None, "header is None"
        assert statement.item_config.header.badges is not None, "badges is None"
        assert len(statement.item_config.header.badges) == 2, f"Expected 2 badges, got {len(statement.item_config.header.badges)}"
        print(f"   Badges: {len(statement.item_config.header.badges)}")
        
        badge1 = statement.item_config.header.badges[0]
        assert badge1.field == "type"
        assert badge1.style == "type_badge"
        assert badge1.transform == "humanize"
        print(f"     Badge 1: field={badge1.field}, style={badge1.style}, transform={badge1.transform}")
        
        badge2 = statement.item_config.header.badges[1]
        assert badge2.field == "status"
        assert badge2.style == "status_badge"
        assert badge2.transform == "humanize"
        print(f"     Badge 2: field={badge2.field}, style={badge2.style}, transform={badge2.transform}")
        
        # Check sections
        assert statement.item_config.sections is not None, "sections is None"
        assert len(statement.item_config.sections) == 3, f"Expected 3 sections, got {len(statement.item_config.sections)}"
        print(f"   Sections: {len(statement.item_config.sections)}")
        
        # Section 1: info_grid
        section1 = statement.item_config.sections[0]
        assert section1.type == "info_grid"
        assert section1.columns == 2
        assert section1.items is not None
        assert len(section1.items) == 4, f"Expected 4 grid items, got {len(section1.items)}"
        print(f"     Section 1: type={section1.type}, columns={section1.columns}, items={len(section1.items)}")
        
        # Check first grid item
        item1 = section1.items[0]
        assert item1.icon == "calendar"
        assert item1.label == "Date & Time"
        assert item1.values is not None
        assert len(item1.values) == 2
        print(f"       Item 1: icon={item1.icon}, label={item1.label}, values={len(item1.values)}")
        
        # Section 2: text_section with condition
        section2 = statement.item_config.sections[1]
        assert section2.type == "text_section"
        assert section2.condition == "reason != null"
        assert section2.style == "reason"
        print(f"     Section 2: type={section2.type}, condition={section2.condition}")
        
        # Section 3: text_section with condition and icon
        section3 = statement.item_config.sections[2]
        assert section3.type == "text_section"
        assert section3.condition == "instructions != null"
        assert section3.icon == "info-circle"
        print(f"     Section 3: type={section3.type}, condition={section3.condition}, icon={section3.icon}")
        
        # Check actions
        assert statement.item_config.actions is not None, "actions is None"
        assert len(statement.item_config.actions) == 3, f"Expected 3 actions, got {len(statement.item_config.actions)}"
        print(f"   Actions: {len(statement.item_config.actions)}")
        
        action1 = statement.item_config.actions[0]
        assert action1.label == "Request Reschedule"
        assert action1.icon == "calendar"
        assert action1.style == "secondary"
        assert action1.action == "request_reschedule"
        assert action1.condition == "status == 'confirmed' && days_until > 2"
        print(f"     Action 1: {action1.label}, condition={action1.condition}")
        
        action2 = statement.item_config.actions[1]
        assert action2.label == "Print Confirmation"
        assert action2.condition == "confirmation_number != null"
        print(f"     Action 2: {action2.label}, condition={action2.condition}")
        
        action3 = statement.item_config.actions[2]
        assert action3.label == "Add to Calendar"
        assert action3.condition == "days_until <= 7 && days_until >= 0"
        print(f"     Action 3: {action3.label}, condition={action3.condition}")
        
        # Check footer
        assert statement.item_config.footer is not None, "footer is None"
        assert statement.item_config.footer.condition == "confirmation_number != null"
        assert statement.item_config.footer.text == "Confirmation #: {{ confirmation_number }}"
        assert statement.item_config.footer.style == "confirmation"
        print(f"   Footer: condition={statement.item_config.footer.condition}")
        
        print("\n✅ Hospital-ai appointments syntax: PASSED")
        print("   All complex nested structures parsed correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hospital-ai appointments syntax: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Hospital-AI Patient Appointments Parser Test")
    print("=" * 60)
    print()
    
    success = test_hospital_appointments_syntax()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Complete syntax validation PASSED!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("⚠️  Syntax validation FAILED")
        print("=" * 60)
        sys.exit(1)
