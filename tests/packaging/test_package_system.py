#!/usr/bin/env python3
"""
Simple test script for the namel3ss package system.
"""

import sys
import tempfile
from pathlib import Path
from textwrap import dedent

# Add namel3ss to path
sys.path.insert(0, str(Path(__file__).parent))

from namel3ss.packages.discovery import PackageDiscovery, load_workspace_config
from namel3ss.loader import load_workspace_program, get_workspace_info


def create_test_workspace():
    """Create a test workspace with packages."""
    # Use a fixed temp directory that we can inspect
    import tempfile
    import shutil
    
    tmp_dir = Path(tempfile.gettempdir()) / "namel3ss_test"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    
    workspace = tmp_dir / "test_workspace"
    workspace.mkdir(parents=True)
    
    # Create workspace config
    config_content = dedent("""
        [workspace]
        name = "test-workspace"
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    # Create a simple package
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    pkg_dir = packages_dir / "test_pkg"
    pkg_dir.mkdir()
    
    # Package manifest
    manifest_content = dedent("""
        [package]
        name = "test.package"
        version = "1.0.0"
        description = "Test package"
        
        modules = ["main"]
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Package module
    module_content = dedent("""
        module main
        
        prompt "test_prompt" {
            template: "Hello from test package!"
        }
    """)
    (pkg_dir / "main.ai").write_text(module_content)
    
    # Workspace module
    workspace_module = dedent("""
        module workspace_main
        
        app "TestApp" {
            description: "Test application"
        }
    """)
    (workspace / "main.ai").write_text(workspace_module)
    
    print(f"Created test workspace at: {workspace}")
    return workspace


def test_package_discovery():
    """Test package discovery functionality."""
    print("🔍 Testing package discovery...")
    
    workspace = create_test_workspace()
    
    try:
        # Test workspace config loading
        config = load_workspace_config(workspace)
        print(f"✅ Loaded workspace config: {config.name}")
        
        # Test package discovery
        discovery = PackageDiscovery(workspace, config)
        packages, workspace_modules = discovery.discover_workspace()
        
        print(f"✅ Found {len(packages)} package(s)")
        print(f"✅ Found {len(workspace_modules)} workspace module(s)")
        
        # Verify package
        if 'test.package' in packages:
            pkg = packages['test.package']
            print(f"✅ Package 'test.package' v{pkg.manifest.version}")
            print(f"   - Modules: {list(pkg.modules.keys())}")
        else:
            print("❌ Package 'test.package' not found")
            return False
        
        # Test workspace info
        info = get_workspace_info(workspace)
        print(f"✅ Workspace info: {info['total_packages']} packages, {info['total_modules']} modules")
        
        return True
        
    except Exception as e:
        print(f"❌ Package discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workspace_loading():
    """Test full workspace loading."""
    print("\n📦 Testing workspace loading...")
    
    workspace = create_test_workspace()
    
    try:
        # Load full workspace
        program, packages = load_workspace_program(workspace)
        
        print(f"✅ Loaded program with {len(program.modules)} modules")
        print(f"✅ Found {len(packages)} packages")
        
        # Verify modules
        module_names = {module.name for module in program.modules}
        expected_modules = {'workspace_main', 'test.package::main'}
        
        if expected_modules.issubset(module_names):
            print("✅ All expected modules loaded")
            for name in sorted(module_names):
                print(f"   - {name}")
        else:
            print(f"❌ Missing modules. Expected: {expected_modules}, Got: {module_names}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Workspace loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_example():
    """Test with our real simple package example."""
    print("\n🏗️ Testing with real example...")
    
    examples_path = Path(__file__).parent / "examples" / "packages" / "simple_package"
    
    if not examples_path.exists():
        print(f"❌ Example not found: {examples_path}")
        return False
    
    try:
        # Test workspace info
        info = get_workspace_info(examples_path)
        print(f"✅ Real example info: {info['total_packages']} packages, {info['total_modules']} modules")
        
        # List packages
        for pkg_name, pkg_info in info['packages'].items():
            print(f"   📦 {pkg_name} v{pkg_info['version']} - {pkg_info['module_count']} modules")
        
        return True
        
    except Exception as e:
        print(f"❌ Real example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing namel3ss Module & Package System")
    print("=" * 50)
    
    tests = [
        test_package_discovery,
        test_workspace_loading,
        test_real_example
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Package system is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())