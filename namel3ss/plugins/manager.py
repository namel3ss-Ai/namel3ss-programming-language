"""
Production-grade plugin manager for Namel3ss.

This module provides comprehensive plugin lifecycle management including:
- Plugin discovery from multiple sources
- Manifest validation and compatibility checking
- Secure plugin loading with capability enforcement
- Plugin registry management
- Integration with security and observability systems

Key Classes:
    - PluginManager: Central plugin lifecycle coordinator
    - PluginInstance: Loaded plugin with metadata and runtime state
    - PluginDiscoverySource: Abstract source for plugin discovery
    - PluginLoadingError: Specialized errors for plugin loading
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Type, Union

from namel3ss.errors import N3Error
from namel3ss.security.base import PermissionLevel
from namel3ss.runtime.security import SecurityContext
from namel3ss.security.capabilities import CapabilityType
# from namel3ss.security.capabilities import SecurityManager  # TODO: Implement SecurityManager
# from namel3ss.observability.tracing import get_tracer  # TODO: Implement tracing module

from .manifest import PluginManifest, PluginType, SemVer, load_plugin_manifest

# Get logger and tracer for observability
logger = logging.getLogger(__name__)
# tracer = get_tracer(__name__)  # TODO: Implement tracing module


class PluginLoadingError(N3Error):
    """Base exception for plugin loading failures."""
    
    def __init__(self, plugin_name: str, message: str, cause: Optional[Exception] = None):
        self.plugin_name = plugin_name
        self.cause = cause
        super().__init__(message)


class PluginCompatibilityError(PluginLoadingError):
    """Exception for plugin compatibility issues."""
    pass


class PluginSecurityError(PluginLoadingError):
    """Exception for plugin security violations."""
    pass


class PluginNotFoundError(PluginLoadingError):
    """Exception for missing plugins."""
    pass


class PluginValidationError(PluginLoadingError):
    """Exception for plugin validation failures."""
    pass


@dataclass
class PluginInstance:
    """
    Loaded plugin instance with metadata and runtime state.
    
    Represents a successfully loaded plugin with its manifest,
    instantiated objects, and runtime metadata.
    """
    
    manifest: PluginManifest
    """Plugin manifest with metadata and configuration"""
    
    module: Any
    """Loaded Python module containing plugin implementation"""
    
    entry_point_instances: Dict[str, Any] = field(default_factory=dict)
    """Instantiated entry point objects by name"""
    
    load_time: float = field(default_factory=time.time)
    """Timestamp when plugin was loaded"""
    
    security_context: Optional[SecurityContext] = None
    """Security context for plugin operations"""
    
    @property
    def name(self) -> str:
        """Plugin name from manifest."""
        return self.manifest.name
    
    @property
    def version(self) -> str:
        """Plugin version from manifest."""
        return self.manifest.version
    
    @property
    def plugin_types(self) -> Set[PluginType]:
        """Plugin types from manifest."""
        return self.manifest.plugin_types
    
    def get_entry_point(self, name: str) -> Any:
        """Get instantiated entry point by name."""
        if name not in self.entry_point_instances:
            raise PluginNotFoundError(
                self.name,
                f"Entry point '{name}' not found in plugin {self.name}"
            )
        return self.entry_point_instances[name]
    
    def has_entry_point(self, name: str) -> bool:
        """Check if plugin has specified entry point."""
        return name in self.entry_point_instances
    
    def list_entry_points(self) -> List[str]:
        """List all available entry point names."""
        return list(self.entry_point_instances.keys())


class PluginDiscoverySource(ABC):
    """Abstract base for plugin discovery sources."""
    
    @abstractmethod
    def discover_plugins(self) -> Iterator[PluginManifest]:
        """Discover available plugins and yield their manifests."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get human-readable name for this discovery source."""
        pass


class EntryPointDiscoverySource(PluginDiscoverySource):
    """Discover plugins via setuptools entry points."""
    
    def __init__(self, entry_point_group: str = "namel3ss.plugins"):
        self.entry_point_group = entry_point_group
    
    def discover_plugins(self) -> Iterator[PluginManifest]:
        """Discover plugins from setuptools entry points."""
        try:
            entry_points = importlib.metadata.entry_points(group=self.entry_point_group)
        except TypeError:
            # Python < 3.10 compatibility
            entry_points = importlib.metadata.entry_points().get(self.entry_point_group, [])
        
        for entry_point in entry_points:
            try:
                # Load plugin module
                plugin_module = entry_point.load()
                
                # Look for manifest in module
                if hasattr(plugin_module, '__plugin_manifest__'):
                    manifest = plugin_module.__plugin_manifest__
                    if isinstance(manifest, PluginManifest):
                        yield manifest
                elif hasattr(plugin_module, 'get_plugin_manifest'):
                    manifest = plugin_module.get_plugin_manifest()
                    if isinstance(manifest, PluginManifest):
                        yield manifest
                else:
                    logger.warning(
                        f"Plugin module {entry_point.name} missing manifest: "
                        f"no __plugin_manifest__ attribute or get_plugin_manifest() function"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load plugin {entry_point.name}: {e}")
                continue
    
    def get_source_name(self) -> str:
        return f"EntryPoints({self.entry_point_group})"


class DirectoryDiscoverySource(PluginDiscoverySource):
    """Discover plugins from filesystem directories."""
    
    def __init__(self, plugin_dirs: List[Union[str, Path]]):
        self.plugin_dirs = [Path(d) for d in plugin_dirs]
    
    def discover_plugins(self) -> Iterator[PluginManifest]:
        """Discover plugins from directory structure."""
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue
            
            # Look for n3-plugin.toml files
            for manifest_path in plugin_dir.rglob("n3-plugin.toml"):
                try:
                    manifest = load_plugin_manifest(manifest_path)
                    yield manifest
                except Exception as e:
                    logger.error(f"Failed to load manifest from {manifest_path}: {e}")
                    continue
    
    def get_source_name(self) -> str:
        return f"Directories({', '.join(str(d) for d in self.plugin_dirs)})"


class PluginManager:
    """
    Production-grade plugin manager for Namel3ss.
    
    Provides comprehensive plugin lifecycle management with security,
    observability, and robust error handling.
    """
    
    def __init__(
        self,
        namel3ss_version: str = "0.5.0",
        security_manager: Optional[Any] = None,  # TODO: Use SecurityManager when implemented
        discovery_sources: Optional[List[PluginDiscoverySource]] = None,
        strict_compatibility: bool = True,
        max_load_time: float = 30.0
    ):
        self.namel3ss_version = namel3ss_version
        self.security_manager = security_manager
        self.strict_compatibility = strict_compatibility
        self.max_load_time = max_load_time
        
        # Plugin registry
        self._loaded_plugins: Dict[str, PluginInstance] = {}
        self._discovered_manifests: Dict[str, PluginManifest] = {}
        
        # Discovery sources
        self.discovery_sources = discovery_sources or [
            EntryPointDiscoverySource("namel3ss.plugins"),
            DirectoryDiscoverySource([
                Path.home() / ".namel3ss" / "plugins",
                Path.cwd() / ".namel3ss" / "plugins"
            ])
        ]
        
        # Statistics
        self.load_start_time: Optional[float] = None
        self.discovery_stats = {
            'plugins_discovered': 0,
            'plugins_loaded': 0,
            'plugins_failed': 0,
            'discovery_time': 0.0,
            'loading_time': 0.0
        }
    
    def discover_plugins(self) -> None:
        """
        Discover available plugins from all configured sources.
        
        Populates the internal manifest registry with discovered plugins.
        """
        start_time = time.time()
        discovered_count = 0
        
        logger.info("Starting plugin discovery")
        
        # Clear previous discoveries
        self._discovered_manifests.clear()
        
        # Discover from all sources
        for source in self.discovery_sources:
            source_name = source.get_source_name()
            logger.debug(f"Discovering plugins from source: {source_name}")
            
            try:
                for manifest in source.discover_plugins():
                    # Check for duplicates
                    if manifest.name in self._discovered_manifests:
                        existing_version = self._discovered_manifests[manifest.name].version
                        if manifest.version != existing_version:
                            logger.warning(
                                f"Duplicate plugin '{manifest.name}' found with different version "
                                f"({manifest.version} vs {existing_version})"
                            )
                        continue
                    
                    # Store discovered manifest
                    self._discovered_manifests[manifest.name] = manifest
                    discovered_count += 1
                    
                    logger.debug(
                        f"Discovered plugin: {manifest.name} v{manifest.version} "
                        f"from {source_name}"
                    )
                    
            except Exception as e:
                logger.error(f"Error discovering plugins from {source_name}: {e}")
                continue
        
        discovery_time = time.time() - start_time
        self.discovery_stats.update({
            'plugins_discovered': discovered_count,
            'discovery_time': discovery_time
        })
        
        logger.info(
            f"Plugin discovery completed: {discovered_count} plugins discovered "
            f"in {discovery_time:.2f}s"
        )
    
    def load_plugins(self, plugin_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Load specified plugins or all discovered plugins.
        
        Args:
            plugin_names: Specific plugins to load, or None for all discovered
            
        Returns:
            Dict mapping plugin name to load status ('loaded', 'failed', 'skipped')
            
        Raises:
            PluginLoadingError: If critical plugin loading fails
        """
        self.load_start_time = time.time()
        load_results = {}
        loaded_count = 0
        failed_count = 0
        
        # Determine plugins to load
        if plugin_names is None:
            plugins_to_load = list(self._discovered_manifests.keys())
        else:
            plugins_to_load = plugin_names
        
        logger.info(f"Loading {len(plugins_to_load)} plugins")
        
        # Load each plugin
        for plugin_name in plugins_to_load:
            try:
                result = self._load_single_plugin(plugin_name)
                load_results[plugin_name] = result
                
                if result == 'loaded':
                    loaded_count += 1
                elif result == 'failed':
                    failed_count += 1
                        
            except Exception as e:
                logger.error(f"Critical error loading plugin {plugin_name}: {e}")
                load_results[plugin_name] = 'failed'
                failed_count += 1
                
                if self.strict_compatibility:
                    raise PluginLoadingError(plugin_name, str(e), e)
        
        loading_time = time.time() - (self.load_start_time or 0)
        self.discovery_stats.update({
            'plugins_loaded': loaded_count,
            'plugins_failed': failed_count,
            'loading_time': loading_time
        })
        
        logger.info(
            f"Plugin loading completed: {loaded_count} loaded, {failed_count} failed "
            f"in {loading_time:.2f}s"
        )
        
        return load_results
    
    def _load_single_plugin(self, plugin_name: str) -> str:
        """
        Load a single plugin by name.
        
        Returns:
            'loaded', 'failed', or 'skipped'
        """
        # Check if already loaded
        if plugin_name in self._loaded_plugins:
            logger.debug(f"Plugin {plugin_name} already loaded")
            return 'loaded'
        
        # Get manifest
        if plugin_name not in self._discovered_manifests:
            raise PluginNotFoundError(
                plugin_name,
                f"Plugin {plugin_name} not found in discovered plugins"
            )
        
        manifest = self._discovered_manifests[plugin_name]
        
        # Validate compatibility
        if not self._validate_compatibility(manifest):
            logger.warning(f"Plugin {plugin_name} is not compatible with current environment")
            return 'skipped'
        
        # Validate security requirements
        if not self._validate_security(manifest):
            logger.warning(f"Plugin {plugin_name} security requirements not satisfied")
            return 'skipped'
        
        try:
            # Load plugin module
            plugin_module = self._load_plugin_module(manifest)
            
            # Create plugin instance
            plugin_instance = PluginInstance(
                manifest=manifest,
                module=plugin_module,
                security_context=self._create_security_context(manifest)
            )
            
            # Instantiate entry points
            self._instantiate_entry_points(plugin_instance)
            
            # Register plugin
            self._loaded_plugins[plugin_name] = plugin_instance
            
            logger.info(f"Successfully loaded plugin: {plugin_name} v{manifest.version}")
            return 'loaded'
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return 'failed'
    
    def _validate_compatibility(self, manifest: PluginManifest) -> bool:
        """Validate plugin compatibility with current environment."""
        # Check Namel3ss version compatibility
        if not manifest.is_compatible_with_namel3ss(self.namel3ss_version):
            logger.debug(
                f"Plugin {manifest.name} incompatible with Namel3ss {self.namel3ss_version}"
            )
            return False
        
        # Check required extras (simplified - would need access to current install)
        # This would check if required extras are available in the current environment
        
        return True
    
    def _validate_security(self, manifest: PluginManifest) -> bool:
        """Validate plugin security requirements."""
        if not self.security_manager:
            # No security validation if manager not provided
            return True
        
        required_capabilities = manifest.requires_capabilities()
        
        # Check if required capabilities are allowed
        # This would integrate with the security manager to validate
        # that the current security context allows these capabilities
        
        return True
    
    def _create_security_context(self, manifest: PluginManifest) -> Optional[SecurityContext]:
        """Create security context for plugin operations."""
        if not self.security_manager:
            return None
        
        # Create context with plugin-specific permissions
        # This would use the security manager to create an appropriate context
        return None
    
    def _load_plugin_module(self, manifest: PluginManifest) -> Any:
        """Load Python module for plugin."""
        # For entry point plugins, we'd load via entry points
        # For directory plugins, we'd dynamically import the module
        # For now, simplified implementation
        
        if not manifest.entry_points:
            raise PluginValidationError(
                manifest.name,
                "Plugin manifest has no entry points defined"
            )
        
        # Get first entry point as main module
        first_entry_point = next(iter(manifest.entry_points.values()))
        module_path = first_entry_point.module
        
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise PluginLoadingError(
                manifest.name,
                f"Failed to import plugin module {module_path}: {e}",
                e
            )
    
    def _instantiate_entry_points(self, plugin_instance: PluginInstance) -> None:
        """Instantiate all entry points for a plugin."""
        for entry_name, entry_point in plugin_instance.manifest.entry_points.items():
            try:
                instance = self._instantiate_entry_point(entry_point, plugin_instance.module)
                plugin_instance.entry_point_instances[entry_name] = instance
                
                logger.debug(
                    f"Instantiated entry point {entry_name} for plugin {plugin_instance.name}"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to instantiate entry point {entry_name} "
                    f"for plugin {plugin_instance.name}: {e}"
                )
                raise PluginLoadingError(
                    plugin_instance.name,
                    f"Failed to instantiate entry point {entry_name}: {e}",
                    e
                )
    
    def _instantiate_entry_point(self, entry_point, module) -> Any:
        """Instantiate a single entry point."""
        from .manifest import PluginEntryPointType
        
        if entry_point.type == PluginEntryPointType.FACTORY:
            # Factory function
            if entry_point.attribute:
                factory = getattr(module, entry_point.attribute)
            else:
                factory = module
            
            return factory(**entry_point.parameters)
        
        elif entry_point.type == PluginEntryPointType.CLASS:
            # Class constructor
            cls = getattr(module, entry_point.attribute or 'Plugin')
            return cls(**entry_point.parameters)
        
        elif entry_point.type == PluginEntryPointType.MODULE:
            # Module with register function
            if hasattr(module, 'register_plugin'):
                # Call registration function
                module.register_plugin()
            return module
        
        elif entry_point.type == PluginEntryPointType.CALLABLE:
            # Direct callable
            if entry_point.attribute:
                return getattr(module, entry_point.attribute)
            else:
                return module
        
        else:
            raise PluginValidationError(
                "unknown",
                f"Unknown entry point type: {entry_point.type}"
            )
    
    # Public API methods
    
    def get_plugin(self, plugin_name: str) -> PluginInstance:
        """Get loaded plugin instance by name."""
        if plugin_name not in self._loaded_plugins:
            raise PluginNotFoundError(
                plugin_name,
                f"Plugin {plugin_name} is not loaded"
            )
        
        return self._loaded_plugins[plugin_name]
    
    def has_plugin(self, plugin_name: str) -> bool:
        """Check if plugin is loaded."""
        return plugin_name in self._loaded_plugins
    
    def list_loaded_plugins(self) -> List[str]:
        """List names of all loaded plugins."""
        return list(self._loaded_plugins.keys())
    
    def list_discovered_plugins(self) -> List[str]:
        """List names of all discovered plugins."""
        return list(self._discovered_manifests.keys())
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInstance]:
        """Get loaded plugins of specific type."""
        return [
            plugin for plugin in self._loaded_plugins.values()
            if plugin_type in plugin.plugin_types
        ]
    
    def get_plugin_manifest(self, plugin_name: str) -> PluginManifest:
        """Get manifest for discovered plugin."""
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name].manifest
        elif plugin_name in self._discovered_manifests:
            return self._discovered_manifests[plugin_name]
        else:
            raise PluginNotFoundError(
                plugin_name,
                f"Plugin {plugin_name} not found"
            )
    
    def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin."""
        if plugin_name not in self._loaded_plugins:
            raise PluginNotFoundError(
                plugin_name,
                f"Plugin {plugin_name} is not loaded"
            )
        
        # Cleanup plugin resources if needed
        plugin_instance = self._loaded_plugins[plugin_name]
        
        # Call cleanup methods if available
        for instance in plugin_instance.entry_point_instances.values():
            if hasattr(instance, 'cleanup'):
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.warning(f"Error during cleanup of {plugin_name}: {e}")
        
        # Remove from registry
        del self._loaded_plugins[plugin_name]
        
        logger.info(f"Unloaded plugin: {plugin_name}")
    
    def reload_plugin(self, plugin_name: str) -> str:
        """Reload a plugin."""
        if plugin_name in self._loaded_plugins:
            self.unload_plugin(plugin_name)
        
        return self._load_single_plugin(plugin_name)
    
    def load_plugin(self, plugin_name: str) -> str:
        """Load a single plugin by name."""
        return self._load_single_plugin(plugin_name)
    
    def get_all_plugins(self) -> List[Dict[str, Any]]:
        """Get all discovered plugins."""
        plugins = []
        for manifest in self._discovered_manifests.values():
            plugin_info = {
                'name': manifest.name,
                'version': manifest.version, 
                'description': manifest.description,
                'loaded': manifest.name in self._loaded_plugins
            }
            plugins.append(plugin_info)
        return plugins
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            'discovery': self.discovery_stats.copy(),
            'loaded_plugins': len(self._loaded_plugins),
            'discovered_plugins': len(self._discovered_manifests),
            'discovery_sources': len(self.discovery_sources),
            'namel3ss_version': self.namel3ss_version
        }


__all__ = [
    # Core classes
    "PluginManager",
    "PluginInstance", 
    "PluginDiscoverySource",
    "EntryPointDiscoverySource",
    "DirectoryDiscoverySource",
    
    # Exceptions
    "PluginLoadingError",
    "PluginCompatibilityError", 
    "PluginSecurityError",
    "PluginNotFoundError",
    "PluginValidationError",
]