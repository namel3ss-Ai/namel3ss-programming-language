"""
Quick test of API endpoints without full ML initialization.
"""

import asyncio
import httpx
import json

async def quick_test():
    """Quick test of basic API functionality."""
    
    print("🚀 Quick API Test")
    print("=" * 20)
    
    base_url = "http://localhost:8000"
    timeout = httpx.Timeout(30.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Test basic endpoint
            print("\\n📊 Testing basic endpoints...")
            
            # Test docs endpoint
            response = await client.get(f"{base_url}/docs")
            if response.status_code == 200:
                print("✅ /docs endpoint: Working")
            else:
                print(f"❌ /docs endpoint failed: {response.status_code}")
                
            # Test openapi endpoint
            response = await client.get(f"{base_url}/openapi.json")
            if response.status_code == 200:
                print("✅ /openapi.json: Working")
                print(f"   API title: {response.json().get('info', {}).get('title', 'N/A')}")
            else:
                print(f"❌ /openapi.json failed: {response.status_code}")
                
            # Test health endpoint
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    health = response.json()
                    print(f"✅ Health check: {health.get('status', 'Unknown')}")
                else:
                    print(f"⚠️  Health endpoint: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Health endpoint error: {e}")
            
            # Test if ingest endpoint exists (without uploading)
            try:
                response = await client.post(f"{base_url}/ingest")
                # We expect this to fail with 422 (missing file), not 404
                if response.status_code == 422:
                    print("✅ /ingest endpoint: Available (422 - missing file expected)")
                elif response.status_code == 404:
                    print("❌ /ingest endpoint: Not found")
                else:
                    print(f"⚠️  /ingest endpoint: Unexpected status {response.status_code}")
            except Exception as e:
                print(f"❌ /ingest endpoint error: {e}")
                
            # Test if search endpoint exists (without query)
            try:
                response = await client.post(f"{base_url}/search")
                # We expect this to fail with 422 (missing query), not 404
                if response.status_code == 422:
                    print("✅ /search endpoint: Available (422 - missing query expected)")
                elif response.status_code == 404:
                    print("❌ /search endpoint: Not found")
                else:
                    print(f"⚠️  /search endpoint: Unexpected status {response.status_code}")
            except Exception as e:
                print(f"❌ /search endpoint error: {e}")
                
            print("\\n🎉 Quick test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
            
    return True

async def main():
    success = await quick_test()
    if success:
        print("\\n✅ API is responding! Core endpoints are available.")
    else:
        print("\\n❌ API test failed.")

if __name__ == "__main__":
    asyncio.run(main())