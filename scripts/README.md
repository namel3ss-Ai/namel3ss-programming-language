# Scripts Directory

This directory contains various utility scripts, demos, and test files organized for easy navigation.

## Structure

```
scripts/
├── demos/           # Demonstration scripts
├── tests/           # Standalone test scripts  
└── utilities/       # Development utilities and tools
```

## Demos (`scripts/demos/`)

Interactive demonstration scripts showing various Namel3ss features:

- `advanced_code_actions_demo.py` - Advanced IDE code actions
- `ai_assistant_comprehensive_demo.py` - Comprehensive AI assistant features
- `ai_assistant_demo.py` - Basic AI assistant functionality
- `lsp_demo.py` - Language Server Protocol features
- `performance_demo.py` - Performance testing and benchmarking
- `simple_ai_demo.py` - Simple AI integration examples
- `simple_lsp_demo.py` - Basic LSP functionality
- `simple_navigation_demo.py` - Navigation features
- `symbol_navigation_demo.py` - Symbol navigation and discovery
- `testing_assistant_demo.py` - Testing workflow assistance

## Tests (`scripts/tests/`)

Standalone test scripts for various components and features:

- `test_*.py` - Individual component tests
- `*_test.py` - Integration and API tests
- Performance profiler and benchmark scripts

## Utilities (`scripts/utilities/`)

Development tools and utilities:

- `parser_optimizer.py` - Parser optimization tools
- `verify_*.py` - Verification and validation scripts
- `setup*.sh` - Setup and installation scripts
- `vscode_integration.py` - VS Code integration utilities

## Usage

### Running Demos
```bash
cd scripts/demos
python simple_ai_demo.py
```

### Running Tests
```bash
cd scripts/tests
python test_integration.py
```

### Using Utilities
```bash
cd scripts/utilities
python parser_optimizer.py
```

## Note

These scripts are for development, testing, and demonstration purposes. For production usage, use the main `namel3ss` CLI tool and the examples in the `examples/` directory.