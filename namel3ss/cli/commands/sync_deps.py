"""
namel3ss sync-deps - Intelligent Dependency Management

Analyzes .ai files to detect feature usage and automatically updates
requirements.txt and package.json with only the necessary dependencies.

Features:
- IR-based feature detection (no guessing)
- Non-destructive updates (preserves user customizations)
- Smart path detection (backend/, frontend/, or root)
- Verbose output showing detected features and changes
"""

import sys
from pathlib import Path
from typing import Optional

from namel3ss.deps import DependencyManager


# =============================================================================
# Output Helper
# =============================================================================

def safe_print(msg: str) -> None:
    """Print message, handling Unicode errors gracefully"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Strip emojis and special characters for Windows compatibility
        print(msg.encode('ascii', 'ignore').decode('ascii'))


def cmd_sync_deps(args) -> int:
    """Execute namel3ss sync-deps command"""
    project_root = Path(args.project_root).resolve()
    preview = args.preview
    verbose = args.verbose
    ai_file = args.file
    list_features = args.list_features
    
    # Handle --list option
    if list_features:
        manager = DependencyManager()
        features = manager.list_available_features()
        
        safe_print("✨ Available Namel3ss Features\n")
        safe_print("=" * 60)
        
        categories = {}
        for feature_id, info in features.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((feature_id, info))
        
        for category in sorted(categories.keys()):
            safe_print(f"\n{category.upper()}")
            safe_print("-" * 60)
            for feature_id, info in sorted(categories[category]):
                safe_print(f"  {feature_id:20} - {info['description']}")
                if info['python_packages']:
                    safe_print(f"    Python: {', '.join(info['python_packages'])}")
                if info['npm_packages']:
                    safe_print(f"    NPM:    {', '.join(info['npm_packages'])}")
        
        return 0
    
    # Validate project root
    if not project_root.exists():
        safe_print(f"❌ Error: Project directory '{project_root}' does not exist")
        return 1
    
    try:
        manager = DependencyManager(verbose=verbose)
        
        # Single file mode
        if ai_file:
            ai_file_path = Path(ai_file).resolve()
            if not ai_file_path.exists():
                safe_print(f"❌ Error: File '{ai_file_path}' does not exist")
                return 1
            
            if not ai_file_path.suffix == '.ai':
                safe_print(f"❌ Error: File '{ai_file_path}' is not a .ai file")
                return 1
            
            safe_print(f"🔍 Analyzing {ai_file_path.name}...")
            result = manager.sync_from_file(ai_file_path, preview=preview)
        
        # Project-wide mode
        else:
            safe_print(f"🔍 Analyzing project: {project_root.name}")
            
            # Check for .ai files
            ai_files = list(project_root.rglob("*.ai"))
            if not ai_files:
                safe_print(f"❌ Error: No .ai files found in '{project_root}'")
                return 1
            
            if preview:
                result = manager.preview_dependencies(project_root)
            else:
                result = manager.sync_project(project_root)
        
        # Display results
        features = result['features']
        added_python = result['added_python']
        added_npm = result['added_npm']
        warnings = result['warnings']
        
        if not features:
            safe_print("\n⚠️  No features detected")
            return 0
        
        safe_print(f"\n✨ Detected {len(features)} feature(s):")
        for feature in sorted(features):
            safe_print(f"   • {feature}")
        
        if added_python or added_npm:
            safe_print("\n📦 Dependencies:")
            if added_python:
                safe_print(f"   Python: +{len(added_python)} package(s)")
                if verbose:
                    for pkg in sorted(added_python):
                        safe_print(f"      • {pkg}")
            if added_npm:
                safe_print(f"   NPM:    +{len(added_npm)} package(s)")
                if verbose:
                    for pkg in sorted(added_npm):
                        safe_print(f"      • {pkg}")
        else:
            safe_print("\n✅ All dependencies up to date")
        
        if warnings:
            safe_print(f"\n⚠️  Warnings:")
            for warning in warnings:
                safe_print(f"   • {warning}")
        
        if preview:
            safe_print("\n🔍 Preview mode - no files were modified")
            safe_print("   Run without --preview to apply changes")
        else:
            safe_print("\n✅ Dependencies synchronized successfully")
            
            # Show what to do next
            if added_python or added_npm:
                safe_print("\n📚 Next steps:")
                if added_python:
                    safe_print("   pip install -r backend/requirements.txt")
                if added_npm:
                    safe_print("   cd frontend && npm install")
        
        return 0
        
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def add_sync_deps_command(subparsers):
    """Add sync-deps command to CLI"""
    sync_parser = subparsers.add_parser(
        'sync-deps',
        help='Synchronize dependencies based on .ai file analysis'
    )
    sync_parser.add_argument(
        'project_root',
        nargs='?',
        default='.',
        help='Project root directory (default: current directory)'
    )
    sync_parser.add_argument(
        '-f', '--file',
        help='Analyze specific .ai file instead of entire project'
    )
    sync_parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview changes without modifying files'
    )
    sync_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    sync_parser.add_argument(
        '--list-features',
        action='store_true',
        help='List all available features and their dependencies'
    )
    sync_parser.set_defaults(func=cmd_sync_deps)
