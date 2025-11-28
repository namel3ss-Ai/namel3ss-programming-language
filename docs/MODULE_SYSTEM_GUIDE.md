# Multi-File Module System Guide

The Namel3ss module system allows you to organize large applications across multiple `.ai` or `.n3` files, with automatic dependency resolution and circular dependency detection.

## Quick Start

### Basic Structure

```
my_project/
├── app/
│   ├── main.ai          # Entry point
│   ├── models.ai        # Data models
│   └── config.ai        # Configuration
└── lib/
    └── utils.ai         # Shared utilities
```

### Module Declaration

Every module should declare its name:

```namel3ss
// app/main.ai
module app.main

import app.models
import lib.utils

app "MyApplication" {
    description: "Multi-file application"
}
```

### Import Statements

```namel3ss
// Import a single module
import app.models

// Import multiple modules
import app.models
import app.config
import lib.utils
```

## Module Resolution

The module system resolves module names to file paths:

- `app.main` → `app/main.ai` or `app/main.n3`
- `app.models.user` → `app/models/user.ai` or `app/models/user.n3`
- `.ai` files are preferred over `.n3` files

## Python API

### Load a Multi-Module Project

```python
from namel3ss.modules.system import load_multi_module_project

# Load project starting from entry module
modules, errors = load_multi_module_project(
    entry_module="app.main",
    project_root="./my_project"
)

if errors:
    for error in errors:
        print(f"Error: {error.message}")
else:
    print(f"Loaded {len(modules)} modules successfully")
    for module in modules:
        print(f"  - {module.name} from {module.path}")
```

### Resolve Module Path

```python
from namel3ss.modules.system import resolve_module

# Find where a module is located
path = resolve_module("app.models", project_root="./my_project")
if path:
    print(f"Module found at: {path}")
```

### Manual Module Loading

```python
from namel3ss.modules.system import ModuleResolver

resolver = ModuleResolver(project_root="./my_project")

# Load a specific module
module_info = resolver.load_module("app.main")

print(f"Module: {module_info.name}")
print(f"Path: {module_info.path}")
print(f"Imports: {module_info.imports}")
print(f"Exports: {list(module_info.exports.keys())}")
```

## Dependency Management

### Circular Dependency Detection

The module system automatically detects circular dependencies:

```namel3ss
// a.ai
module app.a
import app.b  // b imports a → circular!

// b.ai
module app.b
import app.a  // a imports b → circular!
```

**Error:**
```
CircularDependencyError: Circular dependency detected: app.a -> app.b -> app.a
```

### Diamond Dependencies (OK)

Diamond dependencies are NOT circular:

```
    A
   / \
  B   C
   \ /
    D
```

```namel3ss
// a.ai
import app.b
import app.c

// b.ai
import app.d

// c.ai
import app.d

// d.ai (no imports)
```

This structure is valid - D can be imported by both B and C.

### Dependency Ordering

Modules are loaded in topological order (dependencies first):

```python
resolver = ModuleResolver(project_root="./my_project")
resolver.load_module("app.main")

# Get modules in dependency order
order = resolver.get_import_order()
print(order)  # ['lib.utils', 'app.models', 'app.config', 'app.main']
```

## Symbol Exports

Modules automatically export their declarations:

```namel3ss
// models.ai
module app.models

app "Models" {
    dataset users {
        name: text
        age: number
    }
}
```

Access exports from other modules:

```python
resolver = ModuleResolver(project_root="./my_project")
resolver.load_module("app.models")

# Get exported symbol
users_dataset = resolver.get_symbol("app.models", "users")
```

## Error Handling

### Module Not Found

```python
try:
    resolver.load_module("nonexistent.module")
except ModuleSystemError as e:
    print(e.message)  # "Cannot import module 'nonexistent.module' (file not found)"
    print(e.code)     # "MODULE_NOT_FOUND"
```

### Syntax Errors

```python
try:
    resolver.load_module("app.broken")
except ModuleSystemError as e:
    print(e.message)  # "Syntax error in module 'app.broken': ..."
    print(e.code)     # "MODULE_SYNTAX_ERROR"
```

### Circular Dependencies

```python
try:
    resolver.load_module("app.circular")
except CircularDependencyError as e:
    print(e.message)  # "Circular dependency detected: app.a -> app.b -> app.a"
    print(e.code)     # "CIRCULAR_DEPENDENCY"
```

## Best Practices

### 1. Clear Module Hierarchy

```
app/
├── main.ai           # Entry point
├── config.ai         # Configuration
├── models/           # Data models
│   ├── user.ai
│   └── post.ai
├── services/         # Business logic
│   ├── auth.ai
│   └── api.ai
└── utils/            # Utilities
    └── validation.ai
```

### 2. Avoid Circular Dependencies

- Keep dependencies flowing in one direction (e.g., services → models → utils)
- Use dependency injection for cross-module communication
- Extract shared code to a separate module

### 3. Explicit Imports

Always declare imports explicitly:

```namel3ss
// Good
import app.models.user
import app.models.post

// Avoid implicit dependencies
```

### 4. Module Naming Conventions

- Use lowercase names: `app.models.user`
- Use dots for hierarchy: `app.services.auth`
- Match file structure: `app.models.user` → `app/models/user.ai`

## Advanced Usage

### Search Paths

Add additional directories for module resolution:

```python
resolver = ModuleResolver(
    project_root="./my_project",
    search_paths=["./vendor", "./lib"]
)
```

### Module System Builder

For complete project builds:

```python
from namel3ss.modules.system import ModuleSystemBuilder

builder = ModuleSystemBuilder(project_root="./my_project")
modules, errors = builder.build_project("app.main")

if errors:
    for error in errors:
        print(f"[{error.code}] {error.message}")
else:
    print(f"Successfully built project with {len(modules)} modules")
```

### Validate Cross-Module References

```python
resolver = ModuleResolver(project_root="./my_project")
resolver.load_module("app.main")

# Check for invalid references
validation_errors = resolver.validate_cross_module_references()
for error in validation_errors:
    print(f"{error.path}:{error.line} - {error.message}")
```

## Examples

### Example 1: Simple Multi-File App

```namel3ss
// config.ai
module app.config

app "Config" {
    env: {
        API_KEY: env("API_KEY")
    }
}

// models.ai
module app.models
import app.config

app "Models" {
    dataset users {
        name: text
        api_key: ctx.config.env.API_KEY
    }
}

// main.ai
module app.main
import app.models
import app.config

app "MainApp" {
    description: "Multi-module application"
}
```

**Load it:**

```python
modules, errors = load_multi_module_project("app.main", "./project")
# Loads: app.config → app.models → app.main
```

### Example 2: Error Collection

```python
from namel3ss.modules.system import ModuleSystemBuilder

builder = ModuleSystemBuilder(project_root="./project")
modules, errors = builder.build_project("app.main")

# Process all errors
for error in errors:
    print(f"{error.path}:{error.line}:{error.column}")
    print(f"  [{error.code}] {error.message}")
```

## Troubleshooting

### "Module not found"
- Check file exists: `project_root/path/to/module.ai`
- Verify module name matches file path
- Ensure `.ai` or `.n3` extension

### "Circular dependency detected"
- Review import chain in error message
- Restructure modules to break cycle
- Extract shared code to new module

### "Syntax error in module"
- Check syntax in the referenced file
- Run parser directly on file to see full error
- Review line/column in error message

## See Also

- [Static Type Checker Guide](./TYPE_CHECKER_GUIDE.md)
- [Enhanced Expressions Guide](./ENHANCED_EXPRESSIONS_GUIDE.md)
- [Editor API Documentation](./EDITOR_API_GUIDE.md)
