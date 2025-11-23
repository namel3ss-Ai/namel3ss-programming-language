"""
Live API test that connects to running server.
"""

import asyncio
import httpx
import json
import tempfile
from pathlib import Path
import time


async def test_live_api():
    """Test the live API server."""
    
    print("🔍 Testing Live API Server")
    print("=" * 30)
    
    base_url = "http://localhost:8000"
    
    # Wait for server to be fully ready
    print("⏳ Waiting for server to be ready...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(10):
            try:
                response = await client.get(f"{base_url}/docs")
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    break
            except:
                await asyncio.sleep(2)
        else:
            print("❌ Server not responding")
            return False
    
    # Test health endpoint
    print("\\n🏥 Testing health endpoint...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check: {health_data.get('status')}")
                components = health_data.get('components', {})
                for comp, status in components.items():
                    icon = "✅" if status == "ok" else "⚠️" 
                    print(f"   {icon} {comp}: {status}")
            else:
                print(f"⚠️  Health endpoint status: {response.status_code}")
    except Exception as e:
        print(f"❌ Health test error: {e}")
    
    # Create a simple test file
    print("\\n📄 Creating test document...")
    test_content = \"\"\"# Namel3ss Test Document

This is a test document for the multimodal RAG system.

## Features
- Document ingestion
- Vector embeddings
- Semantic search
- Multimodal support

The system can handle text, images, and audio content for comprehensive
information retrieval and question answering.
\"\"\"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    print(f"✅ Test file created: {Path(test_file).name}")
    
    # Test ingest endpoint
    print("\\n📤 Testing /ingest endpoint...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(test_file, 'rb') as file:
                files = {'file': ('test.txt', file, 'text/plain')}
                
                response = await client.post(
                    f"{base_url}/ingest",
                    files=files,
                    params={
                        'extract_images': True,
                        'extract_audio': False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Ingestion successful!")
                    print(f"   Document ID: {result.get('document_id')}")
                    print(f"   Chunks: {result.get('num_chunks')}")
                    print(f"   Modalities: {result.get('modalities')}")
                else:
                    print(f"❌ Ingestion failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Ingest test error: {e}")
        return False
    
    # Test search endpoint
    print("\\n🔍 Testing /search endpoint...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            search_data = {
                "query": "multimodal RAG system features",
                "top_k": 5,
                "enable_hybrid": True,
                "enable_reranking": False,
                "modality": "text"
            }
            
            response = await client.post(
                f"{base_url}/search",
                json=search_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Search successful!")
                print(f"   Query: {result.get('query')}")
                print(f"   Results: {len(result.get('results', []))}")
                
                results = result.get('results', [])
                if results:
                    print("\\n📋 Top results:")
                    for i, res in enumerate(results[:3]):
                        score = res.get('score', 'N/A')
                        content = res.get('payload', {}).get('content', '')[:100]
                        print(f"   {i+1}. Score: {score:.3f}")
                        print(f"      Content: {content}...")
                        
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Search test error: {e}")
        return False
    
    # Cleanup
    try:
        Path(test_file).unlink()
        print("\\n🧹 Cleanup completed")
    except:
        pass
    
    print("\\n🎉 Live API test completed successfully!")
    return True


async def main():
    success = await test_live_api()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\\n✅ All tests passed! API is working correctly.")
    else:
        print("\\n❌ Some tests failed. Check server logs.")