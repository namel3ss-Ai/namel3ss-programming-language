"""CLI commands for module introspection and management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from namel3ss.loader import get_workspace_info, find_workspace_root, load_workspace_program
from namel3ss.packages.discovery import ModuleResolver


def cmd_modules(args: argparse.Namespace) -> int:
    """Handle module management commands."""
    
    if args.module_action == 'list':
        return _cmd_modules_list(args)
    elif args.module_action == 'info':
        return _cmd_modules_info(args)
    elif args.module_action == 'deps':
        return _cmd_modules_deps(args)
    elif args.module_action == 'graph':
        return _cmd_modules_graph(args)
    else:
        print(f"Unknown module action: {args.module_action}")
        return 1


def _cmd_modules_list(args: argparse.Namespace) -> int:
    """List all modules in the workspace."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    try:
        workspace_info = get_workspace_info(workspace_root)
        packages = workspace_info['packages']
        workspace_modules = workspace_info['workspace_modules']
        
        all_modules = {}
        
        # Add workspace modules
        for module_name, module_info in workspace_modules.items():
            all_modules[module_name] = {
                'name': module_name,
                'type': 'workspace',
                'package': None,
                'file_path': module_info['file_path']
            }
        
        # Add package modules
        for package_name, package_info in packages.items():
            for module_name in package_info['modules']:
                qualified_name = f"{package_name}::{module_name}"
                all_modules[qualified_name] = {
                    'name': module_name,
                    'qualified_name': qualified_name,
                    'type': 'package',
                    'package': package_name,
                    'file_path': f"packages/{package_name}/{module_name.replace('.', '/')}.ai"
                }
        
        # Filter by package if specified
        if args.package:
            filtered_modules = {
                name: info for name, info in all_modules.items()
                if info['package'] == args.package
            }
            all_modules = filtered_modules
        
        if args.format == 'json':
            print(json.dumps(all_modules, indent=2))
        else:
            if not all_modules:
                if args.package:
                    print(f"No modules found in package '{args.package}'")
                else:
                    print("No modules found in workspace")
                return 0
            
            print(f"Modules in workspace {workspace_root}:")
            if args.package:
                print(f"(filtered by package: {args.package})")
            print("=" * 50)
            
            # Group by type
            workspace_mods = [m for m in all_modules.values() if m['type'] == 'workspace']
            package_mods = [m for m in all_modules.values() if m['type'] == 'package']
            
            if workspace_mods:
                print("\nWorkspace Modules:")
                for module in sorted(workspace_mods, key=lambda x: x['name']):
                    print(f"  {module['name']}")
                    print(f"    Path: {module['file_path']}")
            
            if package_mods:
                print("\nPackage Modules:")
                current_package = None
                for module in sorted(package_mods, key=lambda x: (x['package'], x['name'])):
                    if module['package'] != current_package:
                        current_package = module['package']
                        print(f"\n  Package: {current_package}")
                    
                    print(f"    {module['name']}")
                    print(f"      Qualified: {module['qualified_name']}")
                    print(f"      Path: {module['file_path']}")
        
        return 0
        
    except Exception as e:
        print(f"Error listing modules: {e}")
        return 1


def _cmd_modules_info(args: argparse.Namespace) -> int:
    """Show detailed information about a specific module."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    module_name = args.module
    
    try:
        # Load full program to get module details
        program, packages = load_workspace_program(workspace_root)
        
        # Find the module
        target_module = None
        for module in program.modules:
            if module.name == module_name or module.name.endswith(f"::{module_name}"):
                target_module = module
                break
        
        if not target_module:
            print(f"Module '{module_name}' not found")
            return 1
        
        # Gather module information
        module_info = {
            'name': target_module.name,
            'path': target_module.path,
            'has_app': target_module.has_explicit_app,
            'imports': [],
            'exports': []
        }
        
        # Get imports
        for import_stmt in target_module.imports:
            if hasattr(import_stmt, 'module_path'):  # UseStatement
                module_info['imports'].append({
                    'type': 'use',
                    'module': import_stmt.module_path,
                    'alias': import_stmt.alias,
                    'items': [{'name': item.name, 'alias': item.alias} 
                             for item in import_stmt.imported_items] if import_stmt.imported_items else None
                })
            else:  # Import
                module_info['imports'].append({
                    'type': 'import',
                    'module': import_stmt.module,
                    'alias': import_stmt.alias,
                    'names': [{'name': name.name, 'alias': name.alias}
                             for name in import_stmt.names] if import_stmt.names else None
                })
        
        # TODO: Get exports information once export system is implemented
        
        if args.format == 'json':
            print(json.dumps(module_info, indent=2))
        else:
            print(f"Module: {module_info['name']}")
            print(f"Path: {module_info['path']}")
            print(f"Has App: {module_info['has_app']}")
            
            if module_info['imports']:
                print(f"\nImports ({len(module_info['imports'])}):")
                for imp in module_info['imports']:
                    if imp['type'] == 'use':
                        print(f"  use {imp['module']}")
                        if imp['alias']:
                            print(f"    as {imp['alias']}")
                        if imp['items']:
                            items = ', '.join(
                                f"{item['name']}" + (f" as {item['alias']}" if item['alias'] else "")
                                for item in imp['items']
                            )
                            print(f"    importing: {items}")
                    else:
                        print(f"  import {imp['module']}")
                        if imp['alias']:
                            print(f"    as {imp['alias']}")
                        if imp['names']:
                            names = ', '.join(
                                f"{name['name']}" + (f" as {name['alias']}" if name['alias'] else "")
                                for name in imp['names']
                            )
                            print(f"    names: {names}")
            else:
                print("\nImports: (none)")
        
        return 0
        
    except Exception as e:
        print(f"Error getting module info: {e}")
        return 1


def _cmd_modules_deps(args: argparse.Namespace) -> int:
    """Show module dependency information."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    try:
        # Load and resolve the full program
        program, packages = load_workspace_program(workspace_root)
        from namel3ss.resolver import resolve_program
        from namel3ss.packages.discovery import ModuleResolver, PackageDiscovery, load_workspace_config
        
        # Set up module resolver
        config = load_workspace_config(workspace_root)
        discovery = PackageDiscovery(workspace_root, config)
        packages_dict, workspace_modules = discovery.discover_workspace()
        module_resolver = ModuleResolver(packages_dict, workspace_modules)
        
        resolved = resolve_program(
            program, 
            packages=packages,
            module_resolver=module_resolver
        )
        
        # Build dependency graph
        dep_graph = {}
        for module_name, resolved_module in resolved.modules.items():
            deps = []
            for import_info in resolved_module.imports:
                deps.append({
                    'module': import_info.target_module,
                    'package': import_info.package_name,
                    'type': 'use' if import_info.is_use_statement else 'import'
                })
            dep_graph[module_name] = deps
        
        if args.format == 'json':
            print(json.dumps(dep_graph, indent=2))
        else:
            print(f"Module dependencies in workspace {workspace_root}:")
            print("=" * 50)
            
            for module_name, dependencies in dep_graph.items():
                print(f"\n{module_name}")
                if dependencies:
                    for dep in dependencies:
                        arrow = "use" if dep['type'] == 'use' else "import"
                        pkg_info = f" [{dep['package']}]" if dep['package'] else ""
                        print(f"  {arrow} → {dep['module']}{pkg_info}")
                else:
                    print("  (no dependencies)")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing module dependencies: {e}")
        return 1


def _cmd_modules_graph(args: argparse.Namespace) -> int:
    """Generate module dependency graph."""
    print("Module dependency graph visualization not yet implemented")
    print("Use 'namel3ss modules deps --format json' for programmatic access to dependency data")
    return 0


def add_modules_command(subparsers) -> None:
    """Add modules command to the CLI parser."""
    modules_parser = subparsers.add_parser(
        'modules',
        aliases=['mod'],
        help='Module introspection and management'
    )
    
    # Create subparser for module actions
    modules_subparsers = modules_parser.add_subparsers(
        dest='module_action',
        help='Module management actions'
    )
    
    # List command
    list_parser = modules_subparsers.add_parser(
        'list',
        help='List all modules in workspace'
    )
    list_parser.add_argument(
        '--package', '-p',
        help='Filter by package name'
    )
    list_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Info command  
    info_parser = modules_subparsers.add_parser(
        'info',
        help='Show detailed module information'
    )
    info_parser.add_argument(
        'module',
        help='Module name to get info for'
    )
    info_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Dependencies command
    deps_parser = modules_subparsers.add_parser(
        'deps',
        help='Show module dependencies'
    )
    deps_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Graph command
    graph_parser = modules_subparsers.add_parser(
        'graph',
        help='Generate module dependency graph'
    )
    
    modules_parser.set_defaults(func=cmd_modules)


__all__ = ['cmd_modules', 'add_modules_command']