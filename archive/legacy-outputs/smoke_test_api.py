"""
Production-like smoke test for Namel3ss Multimodal RAG API.

Tests:
1. Environment setup and dependencies
2. API server startup
3. /ingest endpoint functionality
4. /search endpoint functionality
5. Error handling and edge cases
6. Performance under load
"""

import asyncio
import httpx
import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys
import signal
from contextlib import asynccontextmanager


class APITester:
    """Comprehensive API tester for production-like environment."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.server_process: Optional[subprocess.Popen] = None
        
    async def start_server(self, timeout: int = 30) -> bool:
        """Start the API server and wait for it to be ready."""
        
        print("🚀 Starting API server...")
        
        # Start the server
        self.server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1",
            "--access-log",
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to be ready
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(f"{self.base_url}/docs")
                    if response.status_code == 200:
                        print("✅ API server started successfully!")
                        return True
                except:
                    pass
                await asyncio.sleep(1)
        
        print("❌ API server failed to start within timeout")
        return False
    
    def stop_server(self):
        """Stop the API server gracefully."""
        if self.server_process:
            print("🛑 Stopping API server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("✅ API server stopped")
    
    async def test_health_check(self) -> bool:
        """Test basic health check endpoint."""
        print("\\n🔍 Testing health check...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/")
                
                if response.status_code == 200:
                    print("✅ Health check passed")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_api_docs(self) -> bool:
        """Test API documentation endpoint."""
        print("\\n📚 Testing API documentation...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/docs")
                
                if response.status_code == 200:
                    print("✅ API documentation accessible")
                    return True
                else:
                    print(f"❌ API docs failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ API docs error: {e}")
            return False
    
    def create_test_files(self) -> Dict[str, Path]:
        """Create test files for ingestion."""
        
        test_dir = Path(tempfile.mkdtemp(prefix="namel3ss_test_"))
        
        # Create a simple text file
        text_file = test_dir / "test.txt"
        text_file.write_text("This is a test document for multimodal RAG ingestion. It contains sample text for testing.")
        
        # Create a simple Python file
        py_file = test_dir / "sample.py"
        py_file.write_text('''
def hello_world():
    """A simple test function."""
    print("Hello from Namel3ss RAG system!")
    return "success"

if __name__ == "__main__":
    hello_world()
''')
        
        # Create a JSON file
        json_file = test_dir / "data.json"
        json_file.write_text(json.dumps({
            "title": "Test Dataset",
            "description": "Sample JSON data for RAG testing",
            "items": [
                {"id": 1, "name": "Test Item 1", "value": 100},
                {"id": 2, "name": "Test Item 2", "value": 200}
            ]
        }, indent=2))
        
        return {
            "text": text_file,
            "code": py_file, 
            "data": json_file,
            "dir": test_dir
        }
    
    async def test_ingest_endpoint(self) -> bool:
        """Test document ingestion endpoint."""
        print("\\n📤 Testing /ingest endpoint...")
        
        try:
            # Create test files
            test_files = self.create_test_files()
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Test ingesting a text file
                with open(test_files["text"], "rb") as f:
                    files = {"files": ("test.txt", f, "text/plain")}
                    data = {
                        "metadata": json.dumps({"source": "smoke_test", "type": "text"})
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/ingest",
                        files=files,
                        data=data
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Ingest successful: {result.get('message', 'Unknown')}")
                    print(f"   Processed files: {result.get('processed_files', 0)}")
                    print(f"   Document chunks: {result.get('document_chunks', 0)}")
                    return True
                else:
                    print(f"❌ Ingest failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Ingest error: {e}")
            return False
        finally:
            # Cleanup test files
            import shutil
            if 'test_files' in locals():
                shutil.rmtree(test_files["dir"], ignore_errors=True)
    
    async def test_search_endpoint(self) -> bool:
        """Test search endpoint."""
        print("\\n🔍 Testing /search endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test basic text search
                search_data = {
                    "query": "test document",
                    "limit": 5,
                    "include_metadata": True
                }
                
                response = await client.post(
                    f"{self.base_url}/search",
                    json=search_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Search successful")
                    print(f"   Results found: {len(result.get('results', []))}")
                    
                    # Show first result if available
                    results = result.get('results', [])
                    if results:
                        first_result = results[0]
                        print(f"   Top result score: {first_result.get('score', 'N/A')}")
                        print(f"   Content preview: {first_result.get('content', '')[:100]}...")
                    
                    return True
                else:
                    print(f"❌ Search failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test API error handling."""
        print("\\n🚨 Testing error handling...")
        
        test_results = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test 1: Invalid search query
            try:
                response = await client.post(f"{self.base_url}/search", json={})
                if response.status_code in [400, 422]:  # Expected validation error
                    print("✅ Invalid search query handled correctly")
                    test_results.append(True)
                else:
                    print(f"❌ Invalid search query: unexpected status {response.status_code}")
                    test_results.append(False)
            except Exception as e:
                print(f"❌ Error testing invalid search: {e}")
                test_results.append(False)
            
            # Test 2: Missing ingest data
            try:
                response = await client.post(f"{self.base_url}/ingest")
                if response.status_code in [400, 422]:  # Expected validation error
                    print("✅ Missing ingest data handled correctly")
                    test_results.append(True)
                else:
                    print(f"❌ Missing ingest data: unexpected status {response.status_code}")
                    test_results.append(False)
            except Exception as e:
                print(f"❌ Error testing missing ingest data: {e}")
                test_results.append(False)
            
            # Test 3: Non-existent endpoint
            try:
                response = await client.get(f"{self.base_url}/nonexistent")
                if response.status_code == 404:
                    print("✅ Non-existent endpoint handled correctly")
                    test_results.append(True)
                else:
                    print(f"❌ Non-existent endpoint: unexpected status {response.status_code}")
                    test_results.append(False)
            except Exception as e:
                print(f"❌ Error testing non-existent endpoint: {e}")
                test_results.append(False)
        
        return all(test_results)
    
    async def test_performance(self) -> bool:
        """Basic performance test."""
        print("\\n⚡ Testing basic performance...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test concurrent searches
                search_data = {
                    "query": "performance test",
                    "limit": 3
                }
                
                start_time = time.time()
                
                # Make 5 concurrent requests
                tasks = []
                for i in range(5):
                    task = client.post(f"{self.base_url}/search", json=search_data)
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Check results
                successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                
                print(f"✅ Performance test completed")
                print(f"   {successful}/5 requests successful")
                print(f"   Total time: {elapsed:.2f}s")
                print(f"   Average time per request: {elapsed/5:.2f}s")
                
                return successful >= 3  # At least 3/5 should succeed
                
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False
    
    async def run_smoke_test(self) -> Dict[str, bool]:
        """Run complete smoke test suite."""
        
        print("🧪 Namel3ss Multimodal RAG API - Production Smoke Test")
        print("=" * 60)
        
        results = {}
        
        try:
            # Start server
            if not await self.start_server():
                return {"server_startup": False}
            
            # Wait a moment for server to fully initialize
            await asyncio.sleep(2)
            
            # Run tests
            results["health_check"] = await self.test_health_check()
            results["api_docs"] = await self.test_api_docs()
            results["ingest"] = await self.test_ingest_endpoint()
            results["search"] = await self.test_search_endpoint()
            results["error_handling"] = await self.test_error_handling()
            results["performance"] = await self.test_performance()
            
        except Exception as e:
            print(f"❌ Smoke test error: {e}")
            results["test_suite"] = False
        finally:
            self.stop_server()
        
        # Print summary
        print("\\n" + "=" * 60)
        print("📋 SMOKE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\\n📊 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - API is production ready!")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  Most tests passed - minor issues detected")
        else:
            print("❌ Multiple failures - API needs investigation")
        
        return results


async def main():
    """Run the smoke test."""
    
    # Check if we're in the right directory
    if not Path("api/main.py").exists():
        print("❌ Error: api/main.py not found. Please run from project root.")
        return False
    
    # Verify dependencies
    print("🔍 Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import qdrant_client
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Run smoke test
    tester = APITester()
    results = await tester.run_smoke_test()
    
    # Return success status
    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)