# Quick Reference: namel3ss Module & Package System

## Essential Commands

```bash
# Package management
namel3ss packages list [--json]
namel3ss packages info <package_name> [--json]
namel3ss packages deps <package_name>
namel3ss packages check

# Module management  
namel3ss modules list [--package <pkg>]
namel3ss modules info <module_name>
namel3ss modules deps <module_name>
namel3ss modules graph
```

## File Structure Templates

### Basic Workspace
```
my_project/
├── namel3ss.toml          # Workspace config
├── main.ai                # Main module
└── src/
    └── components.ai      # Additional modules
```

### Package Structure
```
my_package/
├── namel3ss.toml          # Package manifest
├── main.ai                # Package modules
├── utils/
│   └── helpers.ai
└── README.md
```

## Configuration Templates

### Workspace Config (`namel3ss.toml`)
```toml
[workspace]
name = "my-project"
module_paths = ["src", "apps"]
package_paths = ["packages", "vendor"]

[dependencies]
"external.package" = ">=1.0.0"
```

### Package Manifest (`namel3ss.toml`)
```toml
[package]
name = "company.package"
version = "1.0.0" 
description = "Package description"

[dependencies]
"other.package" = "^2.0.0"

modules = ["main", "utils.helpers"]
```

## Import Syntax

```namel3ss
# Basic imports
use module_name
use package::module
use package::module as alias

# Hierarchical modules
module company.feature.component
use company.core::utils.validation
```

## Version Constraints

| Pattern | Example | Meaning |
|---------|---------|---------|
| `==` | `==1.2.3` | Exact version |
| `>=` | `>=1.0.0` | Minimum version |
| `^` | `^1.2.0` | Compatible (1.2.0 ≤ v < 2.0.0) |
| `~=` | `~=1.2.0` | Patch level (~1.2.0) |
| `,` | `>=1.0,<2.0` | Range |

## Common Patterns

### Module Declaration
```namel3ss
module customer.support

prompt "greet" {
    template: "Hello {{name}}"
}

export { "greet" }
```

### Cross-Package Import
```namel3ss
module main

use company.core::auth
use external.charts::bar_chart as chart

app "MyApp" {
    description: "App using multiple packages"
}
```

### Package Dependencies
```toml
[dependencies]
"company.core" = ">=2.0.0"    # Stable API
"experimental" = "==1.0.0"    # Pin experimental
"utilities" = "^1.5.0"        # Compatible updates
```

## Best Practices Checklist

- ✅ Use semantic versioning (`MAJOR.MINOR.PATCH`)
- ✅ Clear, hierarchical module names
- ✅ Minimal, necessary dependencies only
- ✅ Export only public interfaces
- ✅ Document packages with descriptions
- ✅ Regular dependency checks (`packages check`)
- ✅ Avoid circular dependencies
- ✅ Organize by functionality, not file type

## Troubleshooting

### Common Issues

**Package not found:**
```bash
namel3ss packages check  # Verify package discovery
```

**Circular dependencies:**
```bash
namel3ss packages deps <package>  # Check dependency chain
```

**Version conflicts:**
```bash
namel3ss packages list  # Check installed versions
```

**Module import errors:**
```bash
namel3ss modules deps <module>  # Verify module dependencies
```

### Debug Commands

```bash
# Workspace info
namel3ss packages list --json | jq '.'

# Full dependency graph
namel3ss modules graph --format dot | dot -Tpng > deps.png

# Verbose checking
namel3ss packages check --verbose
```

## Migration Cheatsheet

### From Simple to Package
1. Create package directory
2. Add `namel3ss.toml` manifest
3. Move modules to package
4. Update imports: `module` → `package::module`
5. Test with `packages check`

### Adding Dependencies
1. Identify required packages
2. Add to manifest `[dependencies]`
3. Update imports to use package syntax
4. Verify with `packages deps`

## Quick Start

```bash
# 1. Create workspace
mkdir my_project && cd my_project

# 2. Initialize workspace
cat > namel3ss.toml << EOF
[workspace]
name = "my-project"
package_paths = ["packages"]
EOF

# 3. Create main module
cat > main.ai << EOF
module main

app "MyApp" {
    description: "My AI application"
}
EOF

# 4. Verify setup
namel3ss packages check
namel3ss modules list
```