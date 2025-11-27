"""Script to fix outdated test files to use current Parser API."""

import re

def fix_data_display_tests():
    """Fix tests/parser/test_data_display_components.py"""
    
    with open('tests/parser/test_data_display_components.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix Parser() instantiation
    content = re.sub(
        r"parser = Parser\(\)\s+app = parser\.parse\(source\)",
        "parser = Parser(source)\n    module = parser.parse()\n    app = module.body[0]",
        content
    )
    
    # Fix page declarations: page test: -> page "Test" at "/test":
    content = re.sub(
        r'page test:\s+path: "/test"\s+title: "Test"',
        'page "Test" at "/test"',
        content
    )
    
    # Fix dataset declarations ONLY at line start: dataset name: -> dataset "name" from inline:
    content = re.sub(
        r'^dataset (\w+):',
        r'dataset "\1" from inline:',
        content,
        flags=re.MULTILINE
    )
    
    # Fix statement access: app.pages[0].statements[0] -> app.pages[0].body[0]
    content = re.sub(
        r'app\.pages\[0\]\.statements\[0\]',
        'app.pages[0].body[0]',
        content
    )
    content = re.sub(
        r'page\.statements\[0\]',
        'page.body[0]',
        content
    )
    
    # Add app declaration where missing (before dataset or page)
    if 'app "Test App"' not in content:
        content = re.sub(
            r"source = '''(\s+)dataset",
            r"source = '''\napp \"Test App\"\n\1dataset",
            content
        )
        content = re.sub(
            r"source = '''(\s+)page",
            r"source = '''\napp \"Test App\"\n\1page",
            content
        )
    
    with open('tests/parser/test_data_display_components.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fixed test_data_display_components.py")


def fix_layout_tests():
    """Fix tests/parser/test_layout_primitives.py"""
    
    with open('tests/parser/test_layout_primitives.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix page declarations: page test_page: -> page "test_page" at "/test":
    content = re.sub(
        r'page (\w+):\s+path: "([^"]+)"',
        r'page "\1" at "\2"',
        content
    )
    
    # Fix Parser() instantiation - different pattern for layout tests
    content = re.sub(
        r"parser = Parser\(source\)\s+app = parser\.parse\(\)",
        "parser = Parser(source)\n    module = parser.parse()\n    app = module.body[0]",
        content
    )
    
    # Fix app.body[0] -> app.pages[0] for page access
    content = re.sub(
        r'app\.body\[0\]\.body\[0\]',
        'app.pages[0].body[0]',
        content
    )
    
    # Add app declaration where missing
    content = re.sub(
        r'source = """(\s+)page "',
        r'source = """\napp "Test App"\n\1page "',
        content
    )
    
    with open('tests/parser/test_layout_primitives.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fixed test_layout_primitives.py")


if __name__ == '__main__':
    fix_data_display_tests()
    fix_layout_tests()
    print("\nAll tests fixed!")
