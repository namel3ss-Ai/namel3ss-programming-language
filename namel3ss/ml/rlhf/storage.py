"""
RLHF Storage Layer - Checkpoint and artifact management.

This module provides storage abstractions for RLHF training artifacts:
- Local filesystem storage
- S3-compatible storage (AWS S3, MinIO)
- Google Cloud Storage (GCS)
- Azure Blob Storage
- Model registry integration
- Checkpoint versioning and metadata
- Artifact compression and deduplication

Supports multiple storage backends with a unified interface.
"""

import os
import json
import shutil
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from urllib.parse import urlparse

from .errors import RLHFStorageError

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """
    Metadata for a stored artifact (checkpoint, model, dataset).
    """
    artifact_id: str
    artifact_type: str  # "checkpoint", "model", "dataset", "config"
    job_name: str
    algorithm: str
    created_at: datetime
    size_bytes: int
    storage_path: str
    
    # Model information
    base_model: Optional[str] = None
    peft_method: Optional[str] = None
    
    # Training metrics
    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    
    # Version control
    version: str = "1.0.0"
    parent_artifact_id: Optional[str] = None
    
    # Tags and labels
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactMetadata':
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    All storage implementations must support:
    - upload_artifact: Store an artifact
    - download_artifact: Retrieve an artifact
    - delete_artifact: Remove an artifact
    - list_artifacts: List available artifacts
    - get_metadata: Retrieve artifact metadata
    """
    
    @abstractmethod
    def upload_artifact(
        self,
        local_path: str,
        remote_path: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """
        Upload an artifact to storage.
        
        Args:
            local_path: Local file or directory path
            remote_path: Remote storage path
            metadata: Artifact metadata
            
        Returns:
            Remote storage URI
        """
        pass
    
    @abstractmethod
    def download_artifact(
        self,
        remote_path: str,
        local_path: str,
    ) -> str:
        """
        Download an artifact from storage.
        
        Args:
            remote_path: Remote storage path
            local_path: Local destination path
            
        Returns:
            Local path where artifact was saved
        """
        pass
    
    @abstractmethod
    def delete_artifact(self, remote_path: str) -> None:
        """
        Delete an artifact from storage.
        
        Args:
            remote_path: Remote storage path
        """
        pass
    
    @abstractmethod
    def list_artifacts(
        self,
        prefix: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """
        List artifacts in storage.
        
        Args:
            prefix: Path prefix filter
            artifact_type: Artifact type filter
            
        Returns:
            List of artifact metadata
        """
        pass
    
    @abstractmethod
    def get_metadata(self, remote_path: str) -> ArtifactMetadata:
        """
        Get metadata for an artifact.
        
        Args:
            remote_path: Remote storage path
            
        Returns:
            Artifact metadata
        """
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.
    
    Stores artifacts in a local directory with metadata in JSON files.
    Useful for development and single-machine deployments.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize local storage backend.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata directory
        self.metadata_dir = self.base_path / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized local storage at {self.base_path}")
    
    def upload_artifact(
        self,
        local_path: str,
        remote_path: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """Upload artifact to local storage."""
        try:
            local_path_obj = Path(local_path)
            remote_path_obj = self.base_path / remote_path
            
            # Create parent directories
            remote_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy artifact
            if local_path_obj.is_dir():
                if remote_path_obj.exists():
                    shutil.rmtree(remote_path_obj)
                shutil.copytree(local_path_obj, remote_path_obj)
                logger.info(f"Copied directory {local_path} to {remote_path_obj}")
            else:
                shutil.copy2(local_path_obj, remote_path_obj)
                logger.info(f"Copied file {local_path} to {remote_path_obj}")
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{metadata.artifact_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return f"file://{remote_path_obj}"
        
        except Exception as e:
            raise RLHFStorageError(
                f"Failed to upload artifact: {str(e)}",
                code="RLHF050",
                context={
                    "local_path": local_path,
                    "remote_path": remote_path,
                    "error": str(e),
                }
            )
    
    def download_artifact(self, remote_path: str, local_path: str) -> str:
        """Download artifact from local storage."""
        try:
            remote_path_obj = self.base_path / remote_path
            local_path_obj = Path(local_path)
            
            if not remote_path_obj.exists():
                raise RLHFStorageError(
                    f"Artifact not found: {remote_path}",
                    code="RLHF051",
                    context={"remote_path": remote_path}
                )
            
            # Copy artifact
            local_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            if remote_path_obj.is_dir():
                if local_path_obj.exists():
                    shutil.rmtree(local_path_obj)
                shutil.copytree(remote_path_obj, local_path_obj)
            else:
                shutil.copy2(remote_path_obj, local_path_obj)
            
            logger.info(f"Downloaded artifact from {remote_path} to {local_path}")
            return str(local_path_obj)
        
        except RLHFStorageError:
            raise
        except Exception as e:
            raise RLHFStorageError(
                f"Failed to download artifact: {str(e)}",
                code="RLHF051",
                context={
                    "remote_path": remote_path,
                    "local_path": local_path,
                    "error": str(e),
                }
            )
    
    def delete_artifact(self, remote_path: str) -> None:
        """Delete artifact from local storage."""
        try:
            remote_path_obj = self.base_path / remote_path
            
            if remote_path_obj.is_dir():
                shutil.rmtree(remote_path_obj)
            elif remote_path_obj.is_file():
                remote_path_obj.unlink()
            
            logger.info(f"Deleted artifact: {remote_path}")
        
        except Exception as e:
            raise RLHFStorageError(
                f"Failed to delete artifact: {str(e)}",
                code="RLHF053",
                context={"remote_path": remote_path, "error": str(e)}
            )
    
    def list_artifacts(
        self,
        prefix: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """List artifacts in local storage."""
        artifacts = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                metadata = ArtifactMetadata.from_dict(metadata_dict)
                
                # Apply filters
                if prefix and not metadata.storage_path.startswith(prefix):
                    continue
                if artifact_type and metadata.artifact_type != artifact_type:
                    continue
                
                artifacts.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        return sorted(artifacts, key=lambda x: x.created_at, reverse=True)
    
    def get_metadata(self, remote_path: str) -> ArtifactMetadata:
        """Get metadata for an artifact."""
        # Search for metadata by storage path
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                metadata = ArtifactMetadata.from_dict(metadata_dict)
                
                if metadata.storage_path == remote_path:
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        raise RLHFStorageError(
            f"Metadata not found for artifact: {remote_path}",
            code="RLHF051",
            context={"remote_path": remote_path}
        )


class S3StorageBackend(StorageBackend):
    """
    S3-compatible storage backend.
    
    Supports AWS S3, MinIO, DigitalOcean Spaces, and other S3-compatible services.
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize S3 storage backend.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all artifacts
            endpoint_url: Custom endpoint URL (for MinIO, etc.)
            region: AWS region
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.ClientError = ClientError
        except ImportError:
            raise RLHFStorageError(
                "boto3 is required for S3 storage. Install with: pip install boto3",
                code="RLHF050",
                context={"backend": "s3"}
            )
        
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        
        # Create S3 client
        s3_config = {}
        if endpoint_url:
            s3_config["endpoint_url"] = endpoint_url
        if region:
            s3_config["region_name"] = region
        
        self.s3_client = boto3.client("s3", **s3_config)
        
        logger.info(f"Initialized S3 storage: bucket={bucket}, prefix={prefix}")
    
    def _get_key(self, path: str) -> str:
        """Get full S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{path.lstrip('/')}"
        return path.lstrip("/")
    
    def upload_artifact(
        self,
        local_path: str,
        remote_path: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """Upload artifact to S3."""
        try:
            local_path_obj = Path(local_path)
            key = self._get_key(remote_path)
            
            # Upload directory or file
            if local_path_obj.is_dir():
                for file_path in local_path_obj.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path_obj)
                        file_key = f"{key}/{relative_path}"
                        self.s3_client.upload_file(str(file_path), self.bucket, file_key)
                        logger.debug(f"Uploaded {file_path} to s3://{self.bucket}/{file_key}")
            else:
                self.s3_client.upload_file(str(local_path_obj), self.bucket, key)
                logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{key}")
            
            # Upload metadata
            metadata_key = f"{key}/.metadata.json"
            metadata_json = json.dumps(metadata.to_dict(), indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=metadata_key,
                Body=metadata_json.encode("utf-8"),
                ContentType="application/json",
            )
            
            return f"s3://{self.bucket}/{key}"
        
        except self.ClientError as e:
            raise RLHFStorageError(
                f"S3 upload failed: {str(e)}",
                code="RLHF050",
                context={
                    "bucket": self.bucket,
                    "key": key,
                    "error": str(e),
                }
            )
    
    def download_artifact(self, remote_path: str, local_path: str) -> str:
        """Download artifact from S3."""
        try:
            key = self._get_key(remote_path)
            local_path_obj = Path(local_path)
            
            # List objects with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=key,
            )
            
            if "Contents" not in response:
                raise RLHFStorageError(
                    f"Artifact not found in S3: {key}",
                    code="RLHF051",
                    context={"bucket": self.bucket, "key": key}
                )
            
            # Download all files
            for obj in response["Contents"]:
                obj_key = obj["Key"]
                if obj_key.endswith("/.metadata.json"):
                    continue  # Skip metadata file
                
                # Determine local file path
                relative_path = obj_key[len(key):].lstrip("/")
                if relative_path:
                    file_path = local_path_obj / relative_path
                else:
                    file_path = local_path_obj
                
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(self.bucket, obj_key, str(file_path))
                logger.debug(f"Downloaded s3://{self.bucket}/{obj_key} to {file_path}")
            
            logger.info(f"Downloaded artifact from s3://{self.bucket}/{key} to {local_path}")
            return str(local_path_obj)
        
        except self.ClientError as e:
            raise RLHFStorageError(
                f"S3 download failed: {str(e)}",
                code="RLHF051",
                context={
                    "bucket": self.bucket,
                    "key": key,
                    "error": str(e),
                }
            )
    
    def delete_artifact(self, remote_path: str) -> None:
        """Delete artifact from S3."""
        try:
            key = self._get_key(remote_path)
            
            # List and delete all objects with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=key,
            )
            
            if "Contents" in response:
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects},
                )
                logger.info(f"Deleted {len(objects)} objects from s3://{self.bucket}/{key}")
        
        except self.ClientError as e:
            raise RLHFStorageError(
                f"S3 delete failed: {str(e)}",
                code="RLHF053",
                context={
                    "bucket": self.bucket,
                    "key": key,
                    "error": str(e),
                }
            )
    
    def list_artifacts(
        self,
        prefix: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """List artifacts in S3."""
        artifacts = []
        
        search_prefix = self._get_key(prefix) if prefix else self.prefix
        
        try:
            # List all metadata files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=search_prefix)
            
            for page in pages:
                if "Contents" not in page:
                    continue
                
                for obj in page["Contents"]:
                    if obj["Key"].endswith("/.metadata.json"):
                        try:
                            response = self.s3_client.get_object(
                                Bucket=self.bucket,
                                Key=obj["Key"],
                            )
                            metadata_dict = json.loads(response["Body"].read())
                            metadata = ArtifactMetadata.from_dict(metadata_dict)
                            
                            if artifact_type and metadata.artifact_type != artifact_type:
                                continue
                            
                            artifacts.append(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata from {obj['Key']}: {e}")
            
            return sorted(artifacts, key=lambda x: x.created_at, reverse=True)
        
        except self.ClientError as e:
            raise RLHFStorageError(
                f"S3 list failed: {str(e)}",
                code="RLHF051",
                context={
                    "bucket": self.bucket,
                    "prefix": search_prefix,
                    "error": str(e),
                }
            )
    
    def get_metadata(self, remote_path: str) -> ArtifactMetadata:
        """Get metadata for an artifact."""
        try:
            key = self._get_key(remote_path)
            metadata_key = f"{key}/.metadata.json"
            
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=metadata_key,
            )
            
            metadata_dict = json.loads(response["Body"].read())
            return ArtifactMetadata.from_dict(metadata_dict)
        
        except self.ClientError as e:
            raise RLHFStorageError(
                f"Failed to get metadata: {str(e)}",
                code="RLHF051",
                context={
                    "bucket": self.bucket,
                    "key": key,
                    "error": str(e),
                }
            )


class StorageManager:
    """
    High-level storage manager with automatic backend selection.
    
    Automatically selects the appropriate storage backend based on URI scheme:
    - file:// or /path -> LocalStorageBackend
    - s3:// -> S3StorageBackend
    - gs:// -> GCSStorageBackend (future)
    - azure:// -> AzureBlobStorageBackend (future)
    """
    
    def __init__(self):
        """Initialize storage manager."""
        self._backends: Dict[str, StorageBackend] = {}
    
    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse storage URI into scheme and path.
        
        Args:
            uri: Storage URI (e.g., "s3://bucket/path", "/local/path")
            
        Returns:
            Tuple of (scheme, path)
        """
        if "://" in uri:
            parsed = urlparse(uri)
            return parsed.scheme, parsed.path.lstrip("/")
        else:
            return "file", uri
    
    def get_backend(self, uri: str) -> StorageBackend:
        """
        Get storage backend for URI.
        
        Args:
            uri: Storage URI
            
        Returns:
            Storage backend instance
        """
        scheme, _ = self._parse_uri(uri)
        
        if scheme == "file":
            if scheme not in self._backends:
                # Use default local storage location
                base_path = os.getenv("RLHF_STORAGE_PATH", "./rlhf_storage")
                self._backends[scheme] = LocalStorageBackend(base_path)
            return self._backends[scheme]
        
        elif scheme == "s3":
            if scheme not in self._backends:
                # Parse S3 URI: s3://bucket/prefix
                parsed = urlparse(uri)
                bucket = parsed.netloc
                endpoint_url = os.getenv("S3_ENDPOINT_URL")
                region = os.getenv("AWS_REGION", "us-east-1")
                
                self._backends[scheme] = S3StorageBackend(
                    bucket=bucket,
                    endpoint_url=endpoint_url,
                    region=region,
                )
            return self._backends[scheme]
        
        else:
            raise RLHFStorageError(
                f"Unsupported storage scheme: {scheme}",
                code="RLHF050",
                context={"uri": uri, "scheme": scheme}
            )
    
    def upload(
        self,
        local_path: str,
        remote_uri: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """
        Upload artifact to storage.
        
        Args:
            local_path: Local file or directory
            remote_uri: Remote storage URI
            metadata: Artifact metadata
            
        Returns:
            Remote storage URI
        """
        backend = self.get_backend(remote_uri)
        _, remote_path = self._parse_uri(remote_uri)
        return backend.upload_artifact(local_path, remote_path, metadata)
    
    def download(self, remote_uri: str, local_path: str) -> str:
        """
        Download artifact from storage.
        
        Args:
            remote_uri: Remote storage URI
            local_path: Local destination
            
        Returns:
            Local path where artifact was saved
        """
        backend = self.get_backend(remote_uri)
        _, remote_path = self._parse_uri(remote_uri)
        return backend.download_artifact(remote_path, local_path)
    
    def delete(self, remote_uri: str) -> None:
        """
        Delete artifact from storage.
        
        Args:
            remote_uri: Remote storage URI
        """
        backend = self.get_backend(remote_uri)
        _, remote_path = self._parse_uri(remote_uri)
        backend.delete_artifact(remote_path)
    
    def list(
        self,
        base_uri: str,
        prefix: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """
        List artifacts in storage.
        
        Args:
            base_uri: Base storage URI
            prefix: Path prefix filter
            artifact_type: Artifact type filter
            
        Returns:
            List of artifact metadata
        """
        backend = self.get_backend(base_uri)
        return backend.list_artifacts(prefix=prefix, artifact_type=artifact_type)
    
    def get_metadata(self, remote_uri: str) -> ArtifactMetadata:
        """
        Get metadata for an artifact.
        
        Args:
            remote_uri: Remote storage URI
            
        Returns:
            Artifact metadata
        """
        backend = self.get_backend(remote_uri)
        _, remote_path = self._parse_uri(remote_uri)
        return backend.get_metadata(remote_path)


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """
    Get global storage manager instance.
    
    Returns:
        StorageManager instance
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
