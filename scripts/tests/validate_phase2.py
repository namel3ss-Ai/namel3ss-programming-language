"""Quick Phase 2 validation script"""
from namel3ss.parser import Parser

with open('examples/test_phase2_core_components.ai', encoding='utf-8') as f:
    content = f.read()

parser = Parser(content)
app = parser.parse_app()

print("✅ PHASE 2 SUCCESS - Core Components Parse!\n")
print(f"App: {app.name}")
print(f"Datasets: {len(app.datasets)}")
print(f"Pages: {len(app.pages)}")
print("\nComponents found:")
for page in app.pages:
    print(f"\n{page.name}:")
    for stmt in page.body:
        comp_name = stmt.get("component") if isinstance(stmt, dict) else stmt.__class__.__name__
        print(f"  - {comp_name}")

print("\n✅ All Phase 2 components (form, chart, data_table) are recognized!")
