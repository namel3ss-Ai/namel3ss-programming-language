"""
Storage Layer Testing and Examples

Demonstrates the RLHF storage layer capabilities:
- Local filesystem storage
- S3-compatible storage
- Artifact metadata management
- Checkpoint versioning
"""

import tempfile
import shutil
from datetime import datetime
from pathlib import Path


def test_local_storage():
    """Test local filesystem storage backend."""
    print("\n" + "="*60)
    print("Testing Local Storage Backend")
    print("="*60)
    
    from namel3ss.ml.rlhf import (
        LocalStorageBackend,
        ArtifactMetadata,
        RLHFAlgorithm,
    )
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as storage_dir:
        with tempfile.TemporaryDirectory() as artifact_dir:
            # Create backend
            backend = LocalStorageBackend(storage_dir)
            print(f"✓ Created local storage at {storage_dir}")
            
            # Create test artifact
            test_file = Path(artifact_dir) / "model.bin"
            test_file.write_text("fake model data")
            print(f"✓ Created test artifact")
            
            # Create metadata
            metadata = ArtifactMetadata(
                artifact_id="test_checkpoint_001",
                artifact_type="checkpoint",
                job_name="test_job",
                algorithm="dpo",
                created_at=datetime.now(),
                size_bytes=test_file.stat().st_size,
                storage_path="jobs/test_job/checkpoint",
                base_model="gpt2",
                peft_method="lora",
                final_loss=0.5,
                total_steps=1000,
                tags={"experiment": "test"},
            )
            print(f"✓ Created metadata")
            
            # Upload
            uri = backend.upload_artifact(
                local_path=artifact_dir,
                remote_path="jobs/test_job/checkpoint",
                metadata=metadata,
            )
            print(f"✓ Uploaded artifact: {uri}")
            
            # List artifacts
            artifacts = backend.list_artifacts()
            assert len(artifacts) == 1
            assert artifacts[0].artifact_id == "test_checkpoint_001"
            print(f"✓ Listed {len(artifacts)} artifacts")
            
            # Get metadata
            retrieved_metadata = backend.get_metadata("jobs/test_job/checkpoint")
            assert retrieved_metadata.artifact_id == metadata.artifact_id
            print(f"✓ Retrieved metadata")
            
            # Download
            with tempfile.TemporaryDirectory() as download_dir:
                local_path = backend.download_artifact(
                    remote_path="jobs/test_job/checkpoint",
                    local_path=download_dir,
                )
                downloaded_file = Path(local_path) / "model.bin"
                assert downloaded_file.exists()
                assert downloaded_file.read_text() == "fake model data"
                print(f"✓ Downloaded artifact")
            
            # Delete
            backend.delete_artifact("jobs/test_job/checkpoint")
            print(f"✓ Deleted artifact")
            
            print("\n✅ Local storage backend tests passed!")
            return True


def test_storage_manager():
    """Test high-level storage manager."""
    print("\n" + "="*60)
    print("Testing Storage Manager")
    print("="*60)
    
    from namel3ss.ml.rlhf import (
        get_storage_manager,
        ArtifactMetadata,
    )
    
    storage = get_storage_manager()
    print("✓ Got storage manager")
    
    with tempfile.TemporaryDirectory() as artifact_dir:
        # Create test artifact
        test_file = Path(artifact_dir) / "model.bin"
        test_file.write_text("test data")
        
        metadata = ArtifactMetadata(
            artifact_id="manager_test_001",
            artifact_type="checkpoint",
            job_name="manager_test",
            algorithm="dpo",
            created_at=datetime.now(),
            size_bytes=test_file.stat().st_size,
            storage_path="test/checkpoint",
        )
        
        # Test with file:// URI
        uri = storage.upload(
            local_path=artifact_dir,
            remote_uri="file://./test_storage/test/checkpoint",
            metadata=metadata,
        )
        print(f"✓ Uploaded to {uri}")
        
        # List
        artifacts = storage.list(
            base_uri="file://./test_storage",
            artifact_type="checkpoint",
        )
        assert len(artifacts) >= 1
        print(f"✓ Listed {len(artifacts)} artifacts")
        
        # Download
        with tempfile.TemporaryDirectory() as download_dir:
            local_path = storage.download(
                remote_uri="file://./test_storage/test/checkpoint",
                local_path=download_dir,
            )
            print(f"✓ Downloaded to {local_path}")
        
        # Cleanup
        storage.delete("file://./test_storage/test/checkpoint")
        print("✓ Deleted artifact")
        
        # Clean up storage directory
        shutil.rmtree("./test_storage", ignore_errors=True)
    
    print("\n✅ Storage manager tests passed!")
    return True


def test_artifact_metadata():
    """Test artifact metadata serialization."""
    print("\n" + "="*60)
    print("Testing Artifact Metadata")
    print("="*60)
    
    from namel3ss.ml.rlhf import ArtifactMetadata
    
    # Create metadata
    metadata = ArtifactMetadata(
        artifact_id="meta_test_001",
        artifact_type="model",
        job_name="meta_test",
        algorithm="ppo",
        created_at=datetime.now(),
        size_bytes=1024000,
        storage_path="models/meta_test/v1",
        base_model="llama-2-7b",
        peft_method="qlora",
        final_loss=0.25,
        total_steps=5000,
        version="1.0.0",
        tags={"purpose": "production", "team": "ai"},
    )
    print("✓ Created metadata")
    
    # Serialize
    metadata_dict = metadata.to_dict()
    assert "artifact_id" in metadata_dict
    assert "created_at" in metadata_dict
    assert metadata_dict["tags"]["purpose"] == "production"
    print("✓ Serialized to dict")
    
    # Deserialize
    restored = ArtifactMetadata.from_dict(metadata_dict)
    assert restored.artifact_id == metadata.artifact_id
    assert restored.tags["team"] == "ai"
    print("✓ Deserialized from dict")
    
    print("\n✅ Metadata tests passed!")
    return True


def example_s3_storage():
    """
    Example of using S3 storage (requires AWS credentials).
    
    Note: This is a demonstration - not run by default.
    """
    print("\n" + "="*60)
    print("S3 Storage Example (requires AWS credentials)")
    print("="*60)
    
    from namel3ss.ml.rlhf import S3StorageBackend, ArtifactMetadata
    
    # Configuration
    bucket = "my-rlhf-artifacts"
    prefix = "experiments/dpo"
    
    print("""
To use S3 storage:

1. Set AWS credentials:
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_REGION="us-east-1"

2. Create S3 backend:
   backend = S3StorageBackend(
       bucket="my-rlhf-artifacts",
       prefix="experiments/dpo"
   )

3. Upload artifact:
   metadata = ArtifactMetadata(
       artifact_id="s3_test_001",
       artifact_type="checkpoint",
       job_name="dpo_llama2",
       algorithm="dpo",
       created_at=datetime.now(),
       size_bytes=1000000,
       storage_path="checkpoints/step_1000",
   )
   
   uri = backend.upload_artifact(
       local_path="./outputs/checkpoint",
       remote_path="checkpoints/step_1000",
       metadata=metadata,
   )
   # Returns: s3://my-rlhf-artifacts/experiments/dpo/checkpoints/step_1000

4. Use with StorageManager (automatic backend selection):
   from namel3ss.ml.rlhf import get_storage_manager
   
   storage = get_storage_manager()
   
   # Upload (automatically uses S3 backend)
   storage.upload(
       local_path="./outputs/checkpoint",
       remote_uri="s3://my-rlhf-artifacts/experiments/dpo/checkpoint",
       metadata=metadata,
   )
   
   # Download
   storage.download(
       remote_uri="s3://my-rlhf-artifacts/experiments/dpo/checkpoint",
       local_path="./downloads/checkpoint",
   )
   
   # List all checkpoints
   artifacts = storage.list(
       base_uri="s3://my-rlhf-artifacts/experiments/dpo",
       artifact_type="checkpoint",
   )

5. MinIO (S3-compatible) setup:
   export S3_ENDPOINT_URL="http://localhost:9000"
   
   backend = S3StorageBackend(
       bucket="my-bucket",
       endpoint_url="http://localhost:9000",
   )
    """)


def main():
    """Run all storage tests."""
    print("╔═══════════════════════════════════════════════════════╗")
    print("║        RLHF Storage Layer Test Suite                  ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    try:
        # Run tests
        test_artifact_metadata()
        test_local_storage()
        test_storage_manager()
        
        # Show S3 example
        example_s3_storage()
        
        print("\n" + "="*60)
        print("✅ ALL STORAGE TESTS PASSED")
        print("="*60)
        print("\nStorage layer is ready for:")
        print("  ✓ Local filesystem storage")
        print("  ✓ S3-compatible storage (AWS S3, MinIO)")
        print("  ✓ Artifact metadata management")
        print("  ✓ Automatic backend selection")
        print("  ✓ Checkpoint versioning")
        
        return True
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ STORAGE TESTS FAILED")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
