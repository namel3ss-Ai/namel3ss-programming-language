# Simple Package Example

This example demonstrates the basics of creating and using a namel3ss package.

## Structure

```
simple_package/
├── namel3ss.toml          # Workspace configuration
├── main.ai                # Main application module
├── packages/
│   └── example_utils/     # Our example package
│       ├── namel3ss.toml  # Package manifest
│       ├── validation.ai  # Validation utilities
│       └── formatting.ai  # Formatting utilities
└── README.md
```

## Key Concepts Demonstrated

- **Package Creation**: How to structure and configure a package
- **Module Organization**: Organizing related functionality into modules
- **Imports**: Using modules from packages with `use` statements
- **Workspace Configuration**: Setting up a workspace to discover packages

## Running the Example

```bash
# Navigate to the example directory
cd examples/packages/simple_package

# List packages in the workspace
namel3ss packages list

# List modules
namel3ss modules list

# Get package information
namel3ss packages info example.utils

# Check dependencies
namel3ss packages check
```

## Package Details

### example.utils Package

A simple utility package containing:

- **validation module**: Input validation utilities
- **formatting module**: Data formatting helpers

### Usage Pattern

The main application imports and uses functionality from the utils package:

```namel3ss
use example.utils::validation
use example.utils::formatting as fmt
```

## Learning Outcomes

After studying this example, you'll understand:

1. How to create a basic package structure
2. How to write a package manifest
3. How to organize modules within a package
4. How to import and use package modules
5. How to configure a workspace for package discovery

## Next Steps

See the `customer_service_system` example for more complex multi-package scenarios.