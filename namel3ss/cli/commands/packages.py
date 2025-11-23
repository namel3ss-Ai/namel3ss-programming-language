"""CLI commands for package management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from namel3ss.loader import get_workspace_info, find_workspace_root, load_workspace_program
from namel3ss.packages.discovery import DependencyResolver
from namel3ss.packages import PackageInfo


def cmd_packages(args: argparse.Namespace) -> int:
    """Handle package management commands."""
    
    if args.package_action == 'list':
        return _cmd_packages_list(args)
    elif args.package_action == 'info':
        return _cmd_packages_info(args)
    elif args.package_action == 'deps':
        return _cmd_packages_deps(args)
    elif args.package_action == 'check':
        return _cmd_packages_check(args)
    elif args.package_action is None:
        # No subcommand provided - show help
        print("usage: namel3ss packages [-h] {list,info,deps,check} ...")
        print("")
        print("Package management operations")
        print("")
        print("positional arguments:")
        print("  {list,info,deps,check}  Package management actions")
        print("    list                  List all packages in workspace")
        print("    info                  Show detailed package information") 
        print("    deps                  Show package dependencies")
        print("    check                 Check package manifests and dependencies")
        print("")
        print("options:")
        print("  -h, --help            show this help message and exit")
        return 1
    else:
        print(f"Unknown package action: {args.package_action}")
        return 1


def _cmd_packages_list(args: argparse.Namespace) -> int:
    """List all packages in the workspace."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    try:
        workspace_info = get_workspace_info(workspace_root)
        packages = workspace_info['packages']
        
        if args.format == 'json':
            print(json.dumps(packages, indent=2))
        else:
            if not packages:
                print("No packages found in workspace")
                return 0
            
            print(f"Packages in workspace {workspace_root}:")
            print("=" * 50)
            
            for package_name, package_info in packages.items():
                print(f"  {package_name}@{package_info['version']}")
                if package_info.get('description'):
                    print(f"    {package_info['description']}")
                print(f"    Modules: {package_info['module_count']} ({', '.join(package_info['modules'][:3])}{'...' if len(package_info['modules']) > 3 else ''})")
                print()
                
        return 0
        
    except Exception as e:
        print(f"Error listing packages: {e}")
        return 1


def _cmd_packages_info(args: argparse.Namespace) -> int:
    """Show detailed information about a specific package."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    package_name = args.package
    
    try:
        workspace_info = get_workspace_info(workspace_root)
        packages = workspace_info['packages']
        
        if package_name not in packages:
            print(f"Package '{package_name}' not found in workspace")
            available = list(packages.keys())
            if available:
                print(f"Available packages: {', '.join(available)}")
            return 1
        
        package_info = packages[package_name]
        
        if args.format == 'json':
            print(json.dumps(package_info, indent=2))
        else:
            print(f"Package: {package_name}")
            print(f"Version: {package_info['version']}")
            if package_info.get('description'):
                print(f"Description: {package_info['description']}")
            print(f"Modules: {package_info['module_count']}")
            
            if package_info['modules']:
                print("\nModules:")
                for module_name in package_info['modules']:
                    print(f"  - {module_name}")
        
        return 0
        
    except Exception as e:
        print(f"Error getting package info: {e}")
        return 1


def _cmd_packages_deps(args: argparse.Namespace) -> int:
    """Show dependency information for the workspace."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    try:
        # Load full workspace with packages
        program, packages = load_workspace_program(workspace_root)
        
        if not packages:
            print("No packages with dependencies found")
            return 0
        
        # Build dependency graph
        resolver = DependencyResolver(packages)
        root_packages = list(packages.keys())
        
        try:
            resolved_order = resolver.resolve_dependencies(root_packages)
            
            if args.format == 'json':
                deps_info = []
                for package in resolved_order:
                    pkg_info = packages[package.manifest.name]
                    deps_info.append({
                        'name': package.manifest.name,
                        'version': package.manifest.version,
                        'dependencies': {
                            name: str(dep.constraint)
                            for name, dep in package.manifest.dependencies.items()
                        }
                    })
                print(json.dumps(deps_info, indent=2))
            else:
                print(f"Dependency resolution for workspace {workspace_root}:")
                print("=" * 50)
                
                for package in resolved_order:
                    print(f"{package.manifest.name}@{package.manifest.version}")
                    if package.manifest.dependencies:
                        for dep_name, dependency in package.manifest.dependencies.items():
                            print(f"  └─ {dependency}")
                    else:
                        print("  └─ (no dependencies)")
                    print()
                
                print(f"\nResolved {len(resolved_order)} packages in dependency order")
        
        except Exception as e:
            print(f"Dependency resolution failed: {e}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error resolving dependencies: {e}")
        return 1


def _cmd_packages_check(args: argparse.Namespace) -> int:
    """Check package manifests and dependencies for issues."""
    workspace_root = find_workspace_root(Path.cwd())
    if not workspace_root:
        print("No namel3ss workspace found (no namel3ss.toml in parent directories)")
        return 1
    
    try:
        workspace_info = get_workspace_info(workspace_root)
        packages = workspace_info['packages']
        
        issues_found = 0
        
        print(f"Checking packages in workspace {workspace_root}...")
        print("=" * 50)
        
        for package_name, package_info in packages.items():
            print(f"Checking package {package_name}@{package_info['version']}...")
            
            # Check for common issues
            if not package_info.get('description'):
                print(f"  ⚠️  Missing description")
                issues_found += 1
            
            if package_info['module_count'] == 0:
                print(f"  ⚠️  No modules found")
                issues_found += 1
            
            # TODO: Add more validation:
            # - Version format validation
            # - Dependency resolution validation
            # - Module file existence checks
            # - Export validation
            
            if issues_found == 0:
                print(f"  ✅ No issues found")
        
        if issues_found > 0:
            print(f"\n{issues_found} issue(s) found")
            return 1
        else:
            print("\n✅ All packages passed validation")
            return 0
        
    except Exception as e:
        print(f"Error checking packages: {e}")
        return 1


def add_packages_command(subparsers) -> None:
    """Add packages command to the CLI parser."""
    packages_parser = subparsers.add_parser(
        'packages',
        aliases=['pkg'],
        help='Package management operations'
    )
    
    # Create subparser for package actions
    packages_subparsers = packages_parser.add_subparsers(
        dest='package_action',
        help='Package management actions'
    )
    
    # List command
    list_parser = packages_subparsers.add_parser(
        'list',
        help='List all packages in workspace'
    )
    list_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Info command
    info_parser = packages_subparsers.add_parser(
        'info',
        help='Show detailed package information'
    )
    info_parser.add_argument(
        'package',
        help='Package name to get info for'
    )
    info_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Dependencies command
    deps_parser = packages_subparsers.add_parser(
        'deps',
        help='Show package dependencies'
    )
    deps_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    # Check command
    check_parser = packages_subparsers.add_parser(
        'check',
        help='Check package manifests and dependencies'
    )
    
    packages_parser.set_defaults(func=cmd_packages)


__all__ = ['cmd_packages', 'add_packages_command']