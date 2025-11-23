"""CLI command for standard library introspection."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from namel3ss.stdlib.typing_enhanced import (
    get_stdlib_checker, ComponentType, StdLibSymbol
)


def cmd_stdlib(args: argparse.Namespace) -> int:
    """Handle stdlib introspection commands."""
    checker = get_stdlib_checker()
    
    if args.stdlib_action == 'list':
        return _cmd_stdlib_list(args, checker)
    elif args.stdlib_action == 'info':
        return _cmd_stdlib_info(args, checker)
    elif args.stdlib_action == 'validate':
        return _cmd_stdlib_validate(args, checker)
    elif args.stdlib_action == 'search':
        return _cmd_stdlib_search(args, checker)
    else:
        print(f"Unknown stdlib action: {args.stdlib_action}")
        return 1


def _cmd_stdlib_list(args: argparse.Namespace, checker) -> int:
    """List available standard library components."""
    component_type = getattr(args, 'component_type', None)
    
    if component_type:
        try:
            comp_type_enum = ComponentType(component_type)
            symbols = checker.list_symbols(comp_type_enum)
        except ValueError:
            print(f"Invalid component type: {component_type}")
            print(f"Valid types: {', '.join(t.value for t in ComponentType)}")
            return 1
    else:
        symbols = checker.list_symbols()
    
    if args.format == 'json':
        _output_json(symbols)
    else:
        _output_table(symbols, component_type)
    
    return 0


def _cmd_stdlib_info(args: argparse.Namespace, checker) -> int:
    """Show detailed information about a stdlib component."""
    component_name = args.component
    symbol = checker.get_symbol(component_name)
    
    if not symbol:
        print(f"Unknown standard library component: {component_name}")
        return 1
    
    if args.format == 'json':
        info = {
            'name': symbol.name,
            'type': symbol.component_type.value,
            'description': symbol.description,
            'spec': _serialize_spec(symbol.value)
        }
        print(json.dumps(info, indent=2))
    else:
        print(f"Component: {symbol.name}")
        print(f"Type: {symbol.component_type.value}")
        print(f"Description: {symbol.description}")
        print()
        _print_spec_details(symbol)
    
    return 0


def _cmd_stdlib_validate(args: argparse.Namespace, checker) -> int:
    """Validate a configuration against stdlib standards."""
    config_file = args.config_file
    component_type = args.component_type
    
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                import json
                config = json.load(f)
            else:
                import toml
                config = toml.load(f)
        
        errors = []
        if component_type == 'memory':
            errors = checker.validate_memory_config(config)
        elif component_type == 'llm':
            errors = checker.validate_llm_config(config)
        elif component_type == 'tool':
            errors = checker.validate_tool_config(config)
        else:
            print(f"Invalid component type: {component_type}")
            return 1
        
        if errors:
            print(f"Validation errors for {component_type} configuration:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print(f"✓ {component_type.title()} configuration is valid")
            return 0
    
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return 1
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return 1


def _cmd_stdlib_search(args: argparse.Namespace, checker) -> int:
    """Search standard library components."""
    query = args.query
    symbols = checker.list_symbols()
    
    # Simple text search in names and descriptions
    matches = []
    query_lower = query.lower()
    
    for symbol in symbols:
        if (query_lower in symbol.name.lower() or 
            query_lower in symbol.description.lower()):
            matches.append(symbol)
    
    if not matches:
        print(f"No components found matching '{query}'")
        return 0
    
    print(f"Found {len(matches)} component(s) matching '{query}':")
    print()
    
    if args.format == 'json':
        _output_json(matches)
    else:
        _output_table(matches)
    
    return 0


def _output_table(symbols: List[StdLibSymbol], filter_type: Optional[str] = None):
    """Output symbols in a formatted table."""
    if not symbols:
        print("No components found.")
        return
    
    # Group by component type
    by_type = {}
    for symbol in symbols:
        type_name = symbol.component_type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(symbol)
    
    for type_name in sorted(by_type.keys()):
        if filter_type and type_name != filter_type:
            continue
            
        print(f"\n{type_name.upper()} COMPONENTS:")
        print("-" * 50)
        
        for symbol in sorted(by_type[type_name], key=lambda s: s.name):
            print(f"  {symbol.name:<20} {symbol.description}")


def _output_json(symbols: List[StdLibSymbol]):
    """Output symbols in JSON format."""
    data = []
    for symbol in symbols:
        data.append({
            'name': symbol.name,
            'type': symbol.component_type.value,
            'description': symbol.description,
            'spec': _serialize_spec(symbol.value)
        })
    
    print(json.dumps(data, indent=2))


def _serialize_spec(spec) -> Dict:
    """Serialize a spec object to a dictionary."""
    if hasattr(spec, '__dict__'):
        result = {}
        for key, value in spec.__dict__.items():
            if not key.startswith('_'):
                if hasattr(value, 'value'):  # Enum
                    result[key] = value.value
                else:
                    result[key] = value
        return result
    else:
        return str(spec)


def _print_spec_details(symbol: StdLibSymbol):
    """Print detailed specification information."""
    spec = symbol.value
    
    if symbol.component_type == ComponentType.MEMORY:
        print("Memory Policy Specification:")
        if hasattr(spec, 'constraints'):
            print(f"  Constraints: {spec.constraints}")
        if hasattr(spec, 'window_size_range'):
            print(f"  Window Size Range: {spec.window_size_range}")
        if hasattr(spec, 'supports_persistence'):
            print(f"  Supports Persistence: {spec.supports_persistence}")
    
    elif symbol.component_type == ComponentType.LLM:
        print("LLM Config Field Specification:")
        if hasattr(spec, 'value_type'):
            print(f"  Type: {spec.value_type}")
        if hasattr(spec, 'default_value'):
            print(f"  Default: {spec.default_value}")
        if hasattr(spec, 'min_value'):
            print(f"  Min Value: {spec.min_value}")
        if hasattr(spec, 'max_value'):
            print(f"  Max Value: {spec.max_value}")
        if hasattr(spec, 'valid_values'):
            print(f"  Valid Values: {spec.valid_values}")
    
    elif symbol.component_type == ComponentType.TOOL:
        print("Tool Category Specification:")
        if hasattr(spec, 'required_fields'):
            print(f"  Required Fields: {', '.join(spec.required_fields)}")
        if hasattr(spec, 'optional_fields'):
            print(f"  Optional Fields: {', '.join(spec.optional_fields)}")
        if hasattr(spec, 'example_config'):
            print(f"  Example Config: {spec.example_config}")


def add_stdlib_command(subparsers) -> None:
    """Add stdlib command to the CLI parser."""
    stdlib_parser = subparsers.add_parser(
        'stdlib',
        help='Standard library introspection and validation'
    )
    
    stdlib_subparsers = stdlib_parser.add_subparsers(
        dest='stdlib_action',
        help='Standard library actions'
    )
    
    # List command
    list_parser = stdlib_subparsers.add_parser(
        'list',
        help='List available standard library components'
    )
    list_parser.add_argument(
        '--type', '-t',
        dest='component_type',
        choices=['memory', 'llm', 'tool'],
        help='Filter by component type'
    )
    list_parser.add_argument(
        '--format', '-f',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    # Info command
    info_parser = stdlib_subparsers.add_parser(
        'info',
        help='Show detailed information about a component'
    )
    info_parser.add_argument(
        'component',
        help='Name of the standard library component'
    )
    info_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    # Validate command
    validate_parser = stdlib_subparsers.add_parser(
        'validate',
        help='Validate a configuration file against stdlib standards'
    )
    validate_parser.add_argument(
        'config_file',
        help='Path to configuration file (.json or .toml)'
    )
    validate_parser.add_argument(
        '--type', '-t',
        dest='component_type',
        choices=['memory', 'llm', 'tool'],
        required=True,
        help='Type of configuration to validate'
    )
    
    # Search command  
    search_parser = stdlib_subparsers.add_parser(
        'search',
        help='Search standard library components'
    )
    search_parser.add_argument(
        'query',
        help='Search query (matches names and descriptions)'
    )
    search_parser.add_argument(
        '--format', '-f',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    stdlib_parser.set_defaults(func=cmd_stdlib)