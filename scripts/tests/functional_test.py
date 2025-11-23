"""
Test actual API functionality with minimal data.
"""

import asyncio
import httpx
import json

async def functional_test():
    """Test actual API functionality."""
    
    print("🔍 Functional API Test")
    print("=" * 25)
    
    base_url = "http://localhost:8000"
    timeout = httpx.Timeout(60.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Test search endpoint with minimal query
            print("\\n🔎 Testing search functionality...")
            search_data = {
                "query": "test search query",
                "top_k": 3,
                "enable_hybrid": False,  # Disable to avoid complexity
                "enable_reranking": False,
                "modality": "text"
            }
            
            response = await client.post(
                f"{base_url}/search",
                json=search_data
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("✅ Search endpoint working!")
                print(f"   Query processed: {result.get('query')}")
                print(f"   Results returned: {len(result.get('results', []))}")
            else:
                print("⚠️  Search returned error (expected with no data):")
                error_detail = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                print(f"   {error_detail}")
                
            # Test ingest with a simple text "file"
            print("\\n📤 Testing ingest functionality...")
            
            # Create a simple text file content
            test_content = "This is a test document for the Namel3ss multimodal RAG system."
            
            files = {'file': ('test.txt', test_content.encode(), 'text/plain')}
            data = {
                'extract_images': False,  # Simplified
                'extract_audio': False
            }
            
            response = await client.post(
                f"{base_url}/ingest",
                files=files,
                data=data
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("✅ Ingest endpoint working!")
                print(f"   Document ID: {result.get('document_id')}")
                print(f"   Chunks processed: {result.get('num_chunks')}")
            else:
                print("⚠️  Ingest returned error (possibly database related):")
                error_detail = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                print(f"   {error_detail}")
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False
            
    print("\\n🏁 Functional test completed!")
    return True

async def main():
    await functional_test()

if __name__ == "__main__":
    asyncio.run(main())