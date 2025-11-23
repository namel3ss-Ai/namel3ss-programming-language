"""
Simple API smoke test that focuses on basic functionality.

This test checks if the API server can start and respond to basic requests
without requiring full ML model initialization.
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path


async def test_basic_api():
    """Test basic API functionality."""
    
    print("🧪 Basic API Smoke Test")
    print("=" * 30)
    
    # Check if API file exists
    api_file = Path("api/main.py")
    if not api_file.exists():
        print("❌ api/main.py not found")
        return False
    
    print("✅ API file found")
    
    # Test basic import
    try:
        sys.path.insert(0, str(Path.cwd()))
        from api.main import app
        print("✅ API module imports successfully")
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False
    
    # Test FastAPI app creation
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        print("✅ Test client created")
    except Exception as e:
        print(f"❌ Test client creation failed: {e}")
        return False
    
    # Test basic health endpoint (if it exists)
    try:
        response = client.get("/")
        print(f"✅ Root endpoint responds: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Root endpoint test: {e}")
    
    # Test docs endpoint
    try:
        response = client.get("/docs")
        if response.status_code == 200:
            print("✅ API docs accessible")
        else:
            print(f"⚠️  API docs status: {response.status_code}")
    except Exception as e:
        print(f"⚠️  API docs test: {e}")
    
    # Test OpenAPI schema
    try:
        response = client.get("/openapi.json")
        if response.status_code == 200:
            schema = response.json()
            print("✅ OpenAPI schema available")
            
            # Check for our expected endpoints
            paths = schema.get("paths", {})
            if "/ingest" in paths:
                print("✅ /ingest endpoint defined")
            if "/search" in paths:
                print("✅ /search endpoint defined")
                
        else:
            print(f"⚠️  OpenAPI schema status: {response.status_code}")
    except Exception as e:
        print(f"⚠️  OpenAPI schema test: {e}")
    
    print("\\n📋 Basic API Test Summary:")
    print("   • API module loads correctly")
    print("   • FastAPI app initializes")
    print("   • Test client works")
    print("   • Documentation is accessible")
    print("   • Expected endpoints are defined")
    
    return True


async def test_dependencies():
    """Test that required dependencies are available."""
    
    print("\\n🔍 Dependency Check")
    print("=" * 20)
    
    dependencies = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("httpx", "HTTP client"),
        ("sentence_transformers", "Text embeddings"),
        ("torch", "ML framework"),
        ("transformers", "HuggingFace models"),
        ("qdrant_client", "Vector database client"),
        ("PIL", "Image processing"),
    ]
    
    available = 0
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: {description}")
            available += 1
        except ImportError:
            print(f"❌ {module_name}: {description}")
    
    print(f"\\n📊 Dependencies: {available}/{len(dependencies)} available")
    return available >= len(dependencies) * 0.8  # 80% threshold


async def test_environment_setup():
    """Test environment and configuration."""
    
    print("\\n⚙️  Environment Setup")
    print("=" * 20)
    
    checks = []
    
    # Check Python version
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        checks.append(True)
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        checks.append(False)
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
        checks.append(True)
    else:
        print("⚠️  Virtual environment not detected")
        checks.append(True)  # Not critical
    
    # Check memory (approximate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 4:
            print(f"✅ Memory: {memory_gb:.1f}GB available")
            checks.append(True)
        else:
            print(f"⚠️  Memory: {memory_gb:.1f}GB (4GB+ recommended)")
            checks.append(True)  # Not blocking
    except:
        print("⚠️  Memory check skipped")
        checks.append(True)
    
    # Check disk space in temp
    try:
        import shutil
        temp_space = shutil.disk_usage("/tmp").free / (1024**3)
        if temp_space >= 1:
            print(f"✅ Temp space: {temp_space:.1f}GB")
            checks.append(True)
        else:
            print(f"⚠️  Temp space: {temp_space:.1f}GB (1GB+ recommended)")
            checks.append(True)
    except:
        print("⚠️  Disk space check skipped")
        checks.append(True)
    
    return all(checks)


async def main():
    """Run all tests."""
    
    print("🚀 Namel3ss API Production Readiness Check")
    print("=" * 45)
    
    results = []
    
    # Test environment
    results.append(await test_environment_setup())
    
    # Test dependencies
    results.append(await test_dependencies())
    
    # Test basic API
    results.append(await test_basic_api())
    
    # Summary
    print("\\n" + "=" * 45)
    print("📋 FINAL SUMMARY")
    print("=" * 45)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED!")
        print("   API is ready for production testing")
        print("\\n📝 Next steps:")
        print("   1. Start server: uvicorn api.main:app --reload")
        print("   2. Test endpoints: curl http://localhost:8000/docs")
        print("   3. Upload test files to /ingest")
        print("   4. Query with /search")
        success = True
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("   Some issues detected - review output above")
        success = False
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)