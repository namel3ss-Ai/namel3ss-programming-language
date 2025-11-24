"""
Plugin registry client for Namel3ss ecosystem.

Provides client interface for discovering, installing, and managing plugins
from remote registries and marketplaces. Supports multiple registry backends.

Key Components:
    - RegistryClient: Main interface for registry operations
    - PluginMetadata: Enhanced plugin metadata from registry
    - RegistryBackend: Pluggable backend for different registry types
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import toml
from pydantic import BaseModel, HttpUrl, Field

from ..plugins.manifest import PluginManifest, SemVer, PluginSecurity
from ..errors import N3Error

logger = logging.getLogger(__name__)


class RegistryError(N3Error):
    """Error in plugin registry operations."""
    
    def __init__(self, message: str, registry_url: str = ""):
        super().__init__(message)
        self.registry_url = registry_url


@dataclass
class PluginDownloadInfo:
    """Information for downloading a plugin."""
    
    download_url: str
    """URL to download plugin package"""
    
    checksum: str
    """SHA256 checksum of plugin package"""
    
    checksum_type: str = "sha256"
    """Type of checksum (sha256, md5, etc.)"""
    
    size_bytes: Optional[int] = None
    """Size of package in bytes"""
    
    content_type: str = "application/zip"
    """MIME type of package"""


@dataclass
class PluginStats:
    """Usage statistics for a plugin."""
    
    download_count: int = 0
    """Total number of downloads"""
    
    weekly_downloads: int = 0
    """Downloads in last week"""
    
    rating: float = 0.0
    """Average user rating (0-5)"""
    
    rating_count: int = 0
    """Number of ratings"""
    
    last_updated: Optional[datetime] = None
    """When plugin was last updated"""


@dataclass
class PluginRegistryMetadata:
    """Enhanced plugin metadata from registry."""
    
    # Core manifest data
    manifest: PluginManifest
    """Plugin manifest"""
    
    # Registry-specific metadata
    registry_id: str
    """Unique ID in registry"""
    
    publisher: str
    """Plugin publisher/organization"""
    
    published_at: datetime
    """When plugin was published"""
    
    download_info: PluginDownloadInfo
    """Download information"""
    
    stats: PluginStats = field(default_factory=PluginStats)
    """Usage statistics"""
    
    tags: Set[str] = field(default_factory=set)
    """Searchable tags"""
    
    categories: Set[str] = field(default_factory=set)
    """Plugin categories"""
    
    homepage_url: Optional[str] = None
    """Plugin homepage/repository URL"""
    
    documentation_url: Optional[str] = None
    """Documentation URL"""
    
    support_url: Optional[str] = None
    """Support/issues URL"""
    
    changelog_url: Optional[str] = None
    """Changelog URL"""
    
    verified: bool = False
    """Whether publisher is verified"""
    
    @property
    def name(self) -> str:
        """Plugin name from manifest."""
        return self.manifest.name
        
    @property
    def version(self) -> SemVer:
        """Plugin version from manifest."""
        return self.manifest.version
        
    @property
    def description(self) -> str:
        """Plugin description from manifest."""
        return self.manifest.description


class RegistrySearchFilter(BaseModel):
    """Search filter for plugin registry."""
    
    query: Optional[str] = None
    """Text search query"""
    
    categories: Optional[List[str]] = None
    """Filter by categories"""
    
    tags: Optional[List[str]] = None
    """Filter by tags"""
    
    plugin_types: Optional[List[str]] = None
    """Filter by plugin types"""
    
    verified_only: bool = False
    """Only return verified plugins"""
    
    min_rating: Optional[float] = None
    """Minimum rating filter"""
    
    min_downloads: Optional[int] = None
    """Minimum download count filter"""
    
    compatibility_target: Optional[str] = None
    """Target compatibility (namel3ss version)"""


class RegistrySearchResult(BaseModel):
    """Result from registry search."""
    
    plugins: List[PluginRegistryMetadata]
    """Found plugins"""
    
    total_count: int
    """Total number of matching plugins"""
    
    page: int = 1
    """Current page number"""
    
    page_size: int = 20
    """Number of results per page"""
    
    has_more: bool = False
    """Whether more results are available"""


class RegistryBackend(ABC):
    """Abstract base for plugin registry backends."""
    
    @abstractmethod
    async def search_plugins(
        self,
        filter_obj: RegistrySearchFilter,
        page: int = 1,
        page_size: int = 20,
    ) -> RegistrySearchResult:
        """Search for plugins in the registry."""
        
    @abstractmethod
    async def get_plugin(self, name: str, version: Optional[str] = None) -> PluginRegistryMetadata:
        """Get plugin metadata by name and version."""
        
    @abstractmethod
    async def get_plugin_versions(self, name: str) -> List[str]:
        """Get all available versions for a plugin."""
        
    @abstractmethod
    async def download_plugin(
        self,
        plugin: PluginRegistryMetadata,
        destination: Path,
    ) -> Path:
        """Download plugin package to destination."""


class HTTPRegistryBackend(RegistryBackend):
    """HTTP-based plugin registry backend."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    async def search_plugins(
        self,
        filter_obj: RegistrySearchFilter,
        page: int = 1,
        page_size: int = 20,
    ) -> RegistrySearchResult:
        """Search plugins via HTTP API."""
        
        search_params = {
            "page": page,
            "page_size": min(page_size, 100),  # Reasonable limit
        }
        
        # Add filter parameters
        if filter_obj.query:
            search_params["query"] = filter_obj.query
        if filter_obj.categories:
            search_params["categories"] = ",".join(filter_obj.categories)
        if filter_obj.tags:
            search_params["tags"] = ",".join(filter_obj.tags)
        if filter_obj.plugin_types:
            search_params["types"] = ",".join(filter_obj.plugin_types)
        if filter_obj.verified_only:
            search_params["verified"] = "true"
        if filter_obj.min_rating:
            search_params["min_rating"] = filter_obj.min_rating
        if filter_obj.min_downloads:
            search_params["min_downloads"] = filter_obj.min_downloads
        if filter_obj.compatibility_target:
            search_params["compatibility"] = filter_obj.compatibility_target
            
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = self._get_headers()
                
                async with session.get(
                    f"{self.base_url}/api/v1/plugins/search",
                    params=search_params,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_search_result(data)
                    
        except aiohttp.ClientError as e:
            raise RegistryError(
                f"Error searching plugins: {e}",
                registry_url=self.base_url,
            ) from e
            
    async def get_plugin(self, name: str, version: Optional[str] = None) -> PluginRegistryMetadata:
        """Get specific plugin metadata."""
        
        url_path = f"/api/v1/plugins/{name}"
        if version:
            url_path += f"/versions/{version}"
            
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = self._get_headers()
                
                async with session.get(
                    f"{self.base_url}{url_path}",
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_plugin_metadata(data)
                    
        except aiohttp.ClientError as e:
            raise RegistryError(
                f"Error getting plugin {name}: {e}",
                registry_url=self.base_url,
            ) from e
            
    async def get_plugin_versions(self, name: str) -> List[str]:
        """Get all versions for a plugin."""
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = self._get_headers()
                
                async with session.get(
                    f"{self.base_url}/api/v1/plugins/{name}/versions",
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return data.get("versions", [])
                    
        except aiohttp.ClientError as e:
            raise RegistryError(
                f"Error getting versions for {name}: {e}",
                registry_url=self.base_url,
            ) from e
            
    async def download_plugin(
        self,
        plugin: PluginRegistryMetadata,
        destination: Path,
    ) -> Path:
        """Download plugin package."""
        
        download_url = plugin.download_info.download_url
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = self._get_headers()
                
                async with session.get(download_url, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Create destination file
                    package_path = destination / f"{plugin.name}-{plugin.version}.zip"
                    
                    # Download with checksum verification
                    hasher = hashlib.sha256()
                    
                    with package_path.open("wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            hasher.update(chunk)
                            f.write(chunk)
                    
                    # Verify checksum
                    actual_checksum = hasher.hexdigest()
                    expected_checksum = plugin.download_info.checksum
                    
                    if actual_checksum != expected_checksum:
                        package_path.unlink()  # Remove corrupted file
                        raise RegistryError(
                            f"Checksum mismatch for {plugin.name}: "
                            f"expected {expected_checksum}, got {actual_checksum}"
                        )
                    
                    logger.info(f"Downloaded {plugin.name} v{plugin.version} to {package_path}")
                    return package_path
                    
        except aiohttp.ClientError as e:
            raise RegistryError(
                f"Error downloading plugin {plugin.name}: {e}",
                registry_url=self.base_url,
            ) from e
            
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            "User-Agent": "Namel3ss-Plugin-Client/1.0",
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
        
    def _parse_search_result(self, data: Dict[str, Any]) -> RegistrySearchResult:
        """Parse search result from API response."""
        plugins = []
        
        for plugin_data in data.get("plugins", []):
            plugin_metadata = self._parse_plugin_metadata(plugin_data)
            plugins.append(plugin_metadata)
            
        return RegistrySearchResult(
            plugins=plugins,
            total_count=data.get("total", len(plugins)),
            page=data.get("page", 1),
            page_size=data.get("page_size", 20),
            has_more=data.get("has_more", False),
        )
        
    def _parse_plugin_metadata(self, data: Dict[str, Any]) -> PluginRegistryMetadata:
        """Parse plugin metadata from API response."""
        # Parse manifest
        manifest_data = data.get("manifest", {})
        manifest = PluginManifest(**manifest_data)
        
        # Parse download info
        download_data = data.get("download", {})
        download_info = PluginDownloadInfo(
            download_url=download_data["url"],
            checksum=download_data["checksum"],
            checksum_type=download_data.get("checksum_type", "sha256"),
            size_bytes=download_data.get("size"),
            content_type=download_data.get("content_type", "application/zip"),
        )
        
        # Parse stats
        stats_data = data.get("stats", {})
        stats = PluginStats(
            download_count=stats_data.get("download_count", 0),
            weekly_downloads=stats_data.get("weekly_downloads", 0),
            rating=stats_data.get("rating", 0.0),
            rating_count=stats_data.get("rating_count", 0),
            last_updated=self._parse_datetime(stats_data.get("last_updated")),
        )
        
        return PluginRegistryMetadata(
            manifest=manifest,
            registry_id=data["id"],
            publisher=data["publisher"],
            published_at=self._parse_datetime(data["published_at"]),
            download_info=download_info,
            stats=stats,
            tags=set(data.get("tags", [])),
            categories=set(data.get("categories", [])),
            homepage_url=data.get("homepage_url"),
            documentation_url=data.get("documentation_url"),
            support_url=data.get("support_url"),
            changelog_url=data.get("changelog_url"),
            verified=data.get("verified", False),
        )
        
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None


class RegistryClient:
    """
    Client for interacting with plugin registries.
    
    Provides high-level interface for discovering, installing, and managing
    plugins from various registry backends.
    """
    
    def __init__(
        self,
        backends: Optional[Dict[str, RegistryBackend]] = None,
        default_backend: str = "official",
        cache_dir: Optional[Path] = None,
    ):
        self.backends = backends or {}
        self.default_backend = default_backend
        self.cache_dir = cache_dir or Path.home() / ".namel3ss" / "plugin_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default backend if not provided
        if not self.backends and default_backend == "official":
            self.add_backend(
                "official",
                HTTPRegistryBackend("https://plugins.namel3ss.org"),
            )
            
    def add_backend(self, name: str, backend: RegistryBackend) -> None:
        """Add a registry backend."""
        self.backends[name] = backend
        logger.info(f"Added registry backend: {name}")
        
    async def search_plugins(
        self,
        query: Optional[str] = None,
        **filter_kwargs,
    ) -> Dict[str, RegistrySearchResult]:
        """Search for plugins across all registries."""
        
        search_filter = RegistrySearchFilter(query=query, **filter_kwargs)
        results = {}
        
        for backend_name, backend in self.backends.items():
            try:
                result = await backend.search_plugins(search_filter)
                results[backend_name] = result
                logger.debug(f"Found {len(result.plugins)} plugins in {backend_name}")
                
            except Exception as e:
                logger.warning(f"Error searching {backend_name}: {e}")
                results[backend_name] = RegistrySearchResult(
                    plugins=[],
                    total_count=0,
                )
                
        return results
        
    async def get_plugin(
        self,
        name: str,
        version: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> PluginRegistryMetadata:
        """Get plugin metadata from specific or default backend."""
        
        backend_name = backend or self.default_backend
        
        if backend_name not in self.backends:
            raise RegistryError(f"Unknown registry backend: {backend_name}")
            
        registry_backend = self.backends[backend_name]
        return await registry_backend.get_plugin(name, version)
        
    async def get_plugin_versions(
        self,
        name: str,
        backend: Optional[str] = None,
    ) -> List[str]:
        """Get all versions for a plugin."""
        
        backend_name = backend or self.default_backend
        
        if backend_name not in self.backends:
            raise RegistryError(f"Unknown registry backend: {backend_name}")
            
        registry_backend = self.backends[backend_name]
        return await registry_backend.get_plugin_versions(name)
        
    async def download_plugin(
        self,
        name: str,
        version: Optional[str] = None,
        backend: Optional[str] = None,
        destination: Optional[Path] = None,
    ) -> Path:
        """Download a plugin package."""
        
        # Get plugin metadata
        plugin = await self.get_plugin(name, version, backend)
        
        # Use cache directory if no destination specified
        download_dir = destination or self.cache_dir
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already cached
        cached_path = download_dir / f"{plugin.name}-{plugin.version}.zip"
        if cached_path.exists():
            logger.info(f"Using cached plugin: {cached_path}")
            return cached_path
            
        # Download from backend
        backend_name = backend or self.default_backend
        registry_backend = self.backends[backend_name]
        
        return await registry_backend.download_plugin(plugin, download_dir)
        
    async def install_plugin(
        self,
        name: str,
        version: Optional[str] = None,
        backend: Optional[str] = None,
        install_dir: Optional[Path] = None,
    ) -> Path:
        """Download and install a plugin."""
        
        # Download plugin package
        package_path = await self.download_plugin(name, version, backend)
        
        # Install directory
        if install_dir is None:
            install_dir = Path.home() / ".namel3ss" / "plugins"
        install_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract package (basic implementation)
        import zipfile
        
        plugin_dir = install_dir / name
        plugin_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(package_path, 'r') as zip_file:
            zip_file.extractall(plugin_dir)
            
        logger.info(f"Installed plugin {name} to {plugin_dir}")
        return plugin_dir
        
    def clear_cache(self) -> None:
        """Clear the plugin cache."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared plugin cache")


# Default registry client instance
_default_client: Optional[RegistryClient] = None


def get_default_registry_client() -> RegistryClient:
    """Get the default registry client instance."""
    global _default_client
    
    if _default_client is None:
        _default_client = RegistryClient()
        
    return _default_client


__all__ = [
    "RegistryClient",
    "RegistryBackend",
    "HTTPRegistryBackend",
    "PluginRegistryMetadata",
    "PluginDownloadInfo",
    "PluginStats",
    "RegistrySearchFilter",
    "RegistrySearchResult",
    "RegistryError",
    "get_default_registry_client",
]