#!/usr/bin/env python3
"""Test knowledge module parsing."""

from namel3ss.lang.grammar import parse_module
from namel3ss.ast import KnowledgeModule, LogicQuery

def test_knowledge_parsing():
    with open('test_knowledge_parsing.n3', 'r') as f:
        code = f.read()
    
    print("Parsing knowledge module test file...")
    try:
        mod = parse_module(code, path="test_knowledge_parsing.n3")
        print(f"✓ Successfully parsed module")
        
        # Find app in body
        from namel3ss.ast import App
        app = None
        for item in mod.body:
            if isinstance(item, App):
                app = item
                break
        
        if app:
            print(f"✓ Found app: {app.name}")
            
            # Check knowledge modules
            if hasattr(app, 'knowledge_modules'):
                print(f"\nKnowledge modules: {len(app.knowledge_modules)}")
                for km in app.knowledge_modules:
                    print(f"  - {km.name}: {len(km.facts)} facts, {len(km.rules)} rules")
                    for fact in km.facts[:3]:  # Show first 3 facts
                        print(f"    fact: {fact}")
                    for rule in km.rules[:3]:  # Show first 3 rules
                        print(f"    rule: {rule}")
            else:
                print("\n⚠ No knowledge_modules field found on App")
            
            # Check queries
            if hasattr(app, 'queries'):
                print(f"\nQueries: {len(app.queries)}")
                for q in app.queries:
                    print(f"  - {q.name}: {len(q.goals)} goals from {len(q.knowledge_sources)} sources")
                    for goal in q.goals[:3]:
                        print(f"    goal: {goal}")
            else:
                print("\n⚠ No queries field found on App")
        else:
            print("⚠ No app found in module")
        
        return True
    except Exception as e:
        print(f"✗ Parse error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_knowledge_parsing()
    exit(0 if success else 1)
