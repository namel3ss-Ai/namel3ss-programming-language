#!/usr/bin/env python3
"""
Verify that all required non-Python assets are included in the package.

This script inspects the built wheel and sdist to ensure that:
1. Frontend JavaScript widget templates are present
2. CRUD service scaffolding templates are present
3. All configuration files, documentation, and Docker files are included

Usage:
    python3 scripts/verify_package_assets.py
"""

import sys
import zipfile
import tarfile
from pathlib import Path


def check_wheel_contents(wheel_path: Path) -> tuple[list[str], list[str]]:
    """Check wheel for required assets and return found/missing lists."""
    required_assets = [
        # Frontend widget JS templates
        "namel3ss/codegen/frontend/templates/widget-core.js",
        "namel3ss/codegen/frontend/templates/widget-rendering.js",
        "namel3ss/codegen/frontend/templates/widget-realtime.js",
        # Project template scaffolding
        "namel3ss/project_templates/crud_service/files/README.md",
        "namel3ss/project_templates/crud_service/files/GETTING_STARTED.md",
        "namel3ss/project_templates/crud_service/files/AUTHENTICATION.md",
        "namel3ss/project_templates/crud_service/files/app.ai",
        "namel3ss/project_templates/crud_service/files/requirements.txt",
        "namel3ss/project_templates/crud_service/files/requirements-dev.txt",
        "namel3ss/project_templates/crud_service/files/pytest.ini",
        "namel3ss/project_templates/crud_service/files/Dockerfile",
        "namel3ss/project_templates/crud_service/files/Makefile",
        "namel3ss/project_templates/crud_service/files/docker-compose.yml",
        "namel3ss/project_templates/crud_service/files/migrations.sql",
        "namel3ss/project_templates/crud_service/files/.env.example",
        "namel3ss/project_templates/crud_service/files/.gitignore",
        "namel3ss/project_templates/crud_service/files/config/.env.example",
    ]
    
    found = []
    missing = []
    
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        file_list = zf.namelist()
        
        for asset in required_assets:
            if asset in file_list:
                found.append(asset)
            else:
                missing.append(asset)
    
    return found, missing


def check_sdist_contents(sdist_path: Path) -> tuple[list[str], list[str]]:
    """Check sdist for required assets and return found/missing lists."""
    required_assets = [
        # Frontend widget JS templates
        "namel3ss/codegen/frontend/templates/widget-core.js",
        "namel3ss/codegen/frontend/templates/widget-rendering.js",
        "namel3ss/codegen/frontend/templates/widget-realtime.js",
        # Project template scaffolding (sample)
        "namel3ss/project_templates/crud_service/files/README.md",
        "namel3ss/project_templates/crud_service/files/Dockerfile",
        "namel3ss/project_templates/crud_service/files/Makefile",
        "namel3ss/project_templates/crud_service/files/.gitignore",
    ]
    
    found = []
    missing = []
    
    with tarfile.open(sdist_path, 'r:gz') as tf:
        # Get all member names and strip the top-level directory
        file_list = [m.name.split('/', 1)[1] if '/' in m.name else m.name 
                     for m in tf.getmembers()]
        
        for asset in required_assets:
            if asset in file_list:
                found.append(asset)
            else:
                missing.append(asset)
    
    return found, missing


def main():
    """Main verification routine."""
    dist_dir = Path(__file__).parent.parent / "dist"
    
    if not dist_dir.exists():
        print("❌ dist/ directory not found. Run 'python3 -m build' first.")
        sys.exit(1)
    
    # Find the latest wheel and sdist
    wheels = sorted(dist_dir.glob("namel3ss-*.whl"))
    sdists = sorted(dist_dir.glob("namel3ss-*.tar.gz"))
    
    if not wheels:
        print("❌ No wheel found in dist/. Run 'python3 -m build' first.")
        sys.exit(1)
    
    if not sdists:
        print("❌ No sdist found in dist/. Run 'python3 -m build' first.")
        sys.exit(1)
    
    wheel_path = wheels[-1]
    sdist_path = sdists[-1]
    
    print(f"🔍 Verifying package assets...")
    print(f"   Wheel: {wheel_path.name}")
    print(f"   Sdist: {sdist_path.name}")
    print()
    
    # Check wheel
    print("📦 Checking wheel contents...")
    wheel_found, wheel_missing = check_wheel_contents(wheel_path)
    
    if wheel_missing:
        print(f"❌ Missing {len(wheel_missing)} assets in wheel:")
        for asset in wheel_missing:
            print(f"   - {asset}")
        print()
    else:
        print(f"✅ All {len(wheel_found)} required assets present in wheel")
        print()
    
    # Check sdist
    print("📦 Checking sdist contents...")
    sdist_found, sdist_missing = check_sdist_contents(sdist_path)
    
    if sdist_missing:
        print(f"❌ Missing {len(sdist_missing)} assets in sdist:")
        for asset in sdist_missing:
            print(f"   - {asset}")
        print()
    else:
        print(f"✅ All {len(sdist_found)} required assets present in sdist")
        print()
    
    # Summary
    if wheel_missing or sdist_missing:
        print("❌ Package verification FAILED")
        print()
        print("💡 To fix:")
        print("   1. Update MANIFEST.in with missing file patterns")
        print("   2. Update pyproject.toml [tool.setuptools.package-data]")
        print("   3. Rebuild: python3 -m build")
        print("   4. Rerun this script")
        sys.exit(1)
    else:
        print("✅ Package verification PASSED")
        print()
        print("All required non-Python assets are properly packaged!")
        sys.exit(0)


if __name__ == "__main__":
    main()
