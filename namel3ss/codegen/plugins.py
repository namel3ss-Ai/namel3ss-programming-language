"""
Plugin-aware code generation for Namel3ss.

Extends the existing codegen system to handle plugin-provided functionality,
including runtime resolution of plugin dependencies and fallback mechanisms.

Key Components:
    - PluginAwareCodegen: Enhanced codegen with plugin resolution
    - PluginResolver: Runtime plugin resolution and loading
    - Plugin template generation for various backends
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..ir.plugins import (
    BackendIRWithPlugins,
    PluginSpec,
    PluginRequirementSpec,
    EnhancedToolSpec,
    EnhancedConnectorSpec,
    validate_plugin_ir,
)
from ..plugins.manager import PluginManager
from ..plugins.manifest import PluginManifest, PluginCompatibility
from ..errors import N3Error

logger = logging.getLogger(__name__)


class PluginResolutionError(N3Error):
    """Error during plugin resolution."""
    
    def __init__(self, message: str, plugin_name: str = ""):
        super().__init__(message)
        self.plugin_name = plugin_name


class PluginAwareCodegen:
    """
    Enhanced code generator with plugin support.
    
    Handles compilation of Namel3ss programs that use plugin-provided
    functionality, including dependency resolution and runtime integration.
    """
    
    def __init__(self, plugin_manager: PluginManager, *, emit_comments: bool = False):
        self.plugin_manager = plugin_manager
        self.resolved_plugins: Dict[str, Any] = {}
        self.emit_comments = emit_comments
        
    def generate_backend(
        self,
        ir: BackendIRWithPlugins,
        target: str = "fastapi",
        output_dir: str = ".",
        include_plugin_setup: bool = True,
    ) -> Dict[str, str]:
        """
        Generate backend code with plugin support.
        
        Args:
            ir: Backend IR with plugin specifications
            target: Backend target (fastapi, serverless, etc.)
            output_dir: Output directory for generated files
            include_plugin_setup: Include plugin initialization code
            
        Returns:
            Dictionary mapping file paths to generated content
        """
        # Validate plugin IR
        validation_errors = validate_plugin_ir(ir)
        if validation_errors:
            raise PluginResolutionError(
                f"Invalid plugin IR: {'; '.join(validation_errors)}"
            )
        
        # Resolve plugin dependencies
        self._resolve_plugin_dependencies(ir.plugin_requirements)
        
        # Generate base backend code
        generated_files = self._generate_base_backend(ir, target, output_dir)
        
        # Add plugin-specific code
        if include_plugin_setup:
            plugin_files = self._generate_plugin_setup(ir, target)
            generated_files.update(plugin_files)
            
        # Generate plugin-aware endpoints
        endpoint_files = self._generate_plugin_aware_endpoints(ir, target)
        generated_files.update(endpoint_files)
        
        return generated_files
        
    def _resolve_plugin_dependencies(self, requirements: PluginRequirementSpec) -> None:
        """Resolve all plugin dependencies."""
        logger.info(f"Resolving {len(requirements.required_plugins)} required plugins")
        
        for plugin_spec in requirements.required_plugins:
            try:
                plugin_instance = self.plugin_manager.load_plugin(
                    plugin_spec.name,
                    version_constraint=plugin_spec.version_constraint,
                )
                self.resolved_plugins[plugin_spec.name] = plugin_instance
                logger.info(f"Resolved required plugin: {plugin_spec.name}")
                
            except Exception as e:
                raise PluginResolutionError(
                    f"Failed to resolve required plugin {plugin_spec.name}: {e}",
                    plugin_name=plugin_spec.name,
                ) from e
        
        # Try to resolve optional plugins
        for plugin_spec in requirements.optional_plugins:
            try:
                plugin_instance = self.plugin_manager.load_plugin(
                    plugin_spec.name,
                    version_constraint=plugin_spec.version_constraint,
                )
                self.resolved_plugins[plugin_spec.name] = plugin_instance
                logger.info(f"Resolved optional plugin: {plugin_spec.name}")
                
            except Exception as e:
                logger.warning(f"Optional plugin {plugin_spec.name} unavailable: {e}")
                
    def _generate_base_backend(
        self,
        ir: BackendIRWithPlugins,
        target: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate base backend code without plugin specifics."""
        # This would integrate with existing codegen
        # For now, return a basic structure
        
        if target == "fastapi":
            return self._generate_fastapi_backend(ir, output_dir)
        elif target == "serverless":
            return self._generate_serverless_backend(ir, output_dir)
        else:
            raise PluginResolutionError(f"Unsupported backend target: {target}")
            
    def _generate_fastapi_backend(
        self,
        ir: BackendIRWithPlugins,
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate FastAPI backend with plugin support."""
        
        main_py = f'''"""
Generated Namel3ss FastAPI backend with plugin support.
Auto-generated - do not edit manually.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from namel3ss.plugins.manager import PluginManager
from namel3ss.runtime.security import SecurityContext, get_security_context

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plugin manager instance
plugin_manager = PluginManager()

# Store for resolved plugins
resolved_plugins: Dict[str, Any] = {{}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with plugin initialization."""
    # Startup: Load plugins
    logger.info("Initializing plugins...")
    await initialize_plugins()
    
    yield
    
    # Shutdown: Clean up plugins
    logger.info("Shutting down plugins...")
    await cleanup_plugins()


async def initialize_plugins():
    """Initialize all required plugins."""
    global resolved_plugins
    
    # Required plugins
{self._generate_plugin_initialization(ir.plugin_requirements.required_plugins)}
    
    # Optional plugins
{self._generate_plugin_initialization(ir.plugin_requirements.optional_plugins, optional=True)}
    
    logger.info(f"Successfully initialized {{len(resolved_plugins)}} plugins")


async def cleanup_plugins():
    """Clean up plugin resources."""
    global resolved_plugins
    
    for plugin_name, plugin_instance in resolved_plugins.items():
        try:
            if hasattr(plugin_instance, 'cleanup'):
                await plugin_instance.cleanup()
            logger.info(f"Cleaned up plugin: {{plugin_name}}")
        except Exception as e:
            logger.error(f"Error cleaning up plugin {{plugin_name}}: {{e}}")
    
    resolved_plugins.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Namel3ss Generated Backend",
    description="Auto-generated backend with plugin support",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_plugin_manager() -> PluginManager:
    """Get the plugin manager instance."""
    return plugin_manager


def get_resolved_plugins() -> Dict[str, Any]:
    """Get resolved plugin instances."""
    return resolved_plugins


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with plugin status."""
    plugin_status = {{}}
    for name, instance in resolved_plugins.items():
        plugin_status[name] = {{
            "loaded": True,
            "type": getattr(instance, '__class__', {{}}).__name__,
        }}
    
    return {{
        "status": "healthy",
        "plugins": plugin_status,
    }}


# Generated endpoint implementations will be added here
{self._generate_endpoint_implementations(ir)}
'''
        
        requirements_txt = f'''# Generated requirements for Namel3ss backend
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Plugin dependencies
{self._generate_plugin_requirements(ir.plugin_requirements)}
'''
        
        dockerfile = f'''# Generated Dockerfile for Namel3ss backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Plugin setup
RUN mkdir -p /app/plugins
ENV NAMEL3SS_PLUGIN_PATH="/app/plugins"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        return {
            f"{output_dir}/main.py": main_py,
            f"{output_dir}/requirements.txt": requirements_txt,
            f"{output_dir}/Dockerfile": dockerfile,
        }
        
    def _generate_serverless_backend(
        self,
        ir: BackendIRWithPlugins,
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate serverless backend (AWS Lambda) with plugin support."""
        
        handler_py = f'''"""
Generated Namel3ss serverless backend with plugin support.
Auto-generated - do not edit manually.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional

from namel3ss.plugins.manager import PluginManager
from namel3ss.runtime.security import SecurityContext

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global plugin manager and resolved plugins
plugin_manager: Optional[PluginManager] = None
resolved_plugins: Dict[str, Any] = {{}}


async def initialize_plugins():
    """Initialize plugins for serverless environment."""
    global plugin_manager, resolved_plugins
    
    if plugin_manager is None:
        plugin_manager = PluginManager()
    
    if not resolved_plugins:
        # Initialize required plugins
{self._generate_plugin_initialization(ir.plugin_requirements.required_plugins, indent="        ")}
        
        # Initialize optional plugins  
{self._generate_plugin_initialization(ir.plugin_requirements.optional_plugins, optional=True, indent="        ")}
        
        logger.info(f"Initialized {{len(resolved_plugins)}} plugins for serverless")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler with plugin support."""
    try:
        # Initialize plugins if needed
        asyncio.run(initialize_plugins())
        
        # Extract request information
        http_method = event.get("httpMethod", "GET")
        path = event.get("path", "/")
        
        # Route request
        response = asyncio.run(handle_request(http_method, path, event))
        
        return {{
            "statusCode": response.get("statusCode", 200),
            "headers": response.get("headers", {{}}),
            "body": json.dumps(response.get("body", {{}})),
        }}
        
    except Exception as e:
        logger.error(f"Error in lambda handler: {{e}}", exc_info=True)
        return {{
            "statusCode": 500,
            "headers": {{"Content-Type": "application/json"}},
            "body": json.dumps({{"error": "Internal server error"}}),
        }}


async def handle_request(method: str, path: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming request with plugin support."""
    
    # Security context
    security_context = SecurityContext(
        user_id=event.get("requestContext", {{}}).get("identity", {{}}).get("userArn"),
        permissions=set(),  # Extract from JWT or API key
    )
    
    # Route to appropriate handler based on path and method
{self._generate_serverless_routing(ir)}
    
    return {{
        "statusCode": 404,
        "body": {{"error": "Not found"}},
    }}


# Generated endpoint handlers
{self._generate_serverless_handlers(ir)}
'''
        
        serverless_yml = f'''# Generated serverless configuration
service: namel3ss-backend

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  environment:
    NAMEL3SS_PLUGIN_PATH: /tmp/plugins

functions:
  api:
    handler: handler.lambda_handler
    events:
      - http:
          path: /{{proxy+}}
          method: ANY
          cors: true
    layers:
      - ${{self:custom.layerArn}}

custom:
  layerArn: !Ref PluginLayer

resources:
  Resources:
    PluginLayer:
      Type: AWS::Lambda::LayerVersion
      Properties:
        LayerName: namel3ss-plugins
        ContentUri: layers/plugins
        CompatibleRuntimes:
          - python3.11

plugins:
  - serverless-python-requirements
'''
        
        return {
            f"{output_dir}/handler.py": handler_py,
            f"{output_dir}/serverless.yml": serverless_yml,
            f"{output_dir}/requirements.txt": self._generate_plugin_requirements(ir.plugin_requirements),
        }
        
    def _generate_plugin_setup(self, ir: BackendIRWithPlugins, target: str) -> Dict[str, str]:
        """Generate plugin setup and configuration files."""
        
        plugin_config_py = f'''"""
Plugin configuration for generated backend.
"""

from typing import Dict, Any
from namel3ss.plugins.manifest import PluginManifest

# Plugin configurations
PLUGIN_CONFIGS: Dict[str, Dict[str, Any]] = {{
{self._generate_plugin_configs(ir)}
}}

# Plugin manifests (for validation)
PLUGIN_MANIFESTS: Dict[str, PluginManifest] = {{}}

def load_plugin_manifests():
    """Load and validate plugin manifests."""
    global PLUGIN_MANIFESTS
    
    # This would be populated at runtime
    # by reading actual plugin manifests
    pass


def get_plugin_config(plugin_name: str) -> Dict[str, Any]:
    """Get configuration for a specific plugin."""
    return PLUGIN_CONFIGS.get(plugin_name, {{}})


def validate_plugin_compatibility():
    """Validate that all plugins are compatible with this backend."""
    # Implementation would check version constraints,
    # capability requirements, etc.
    pass
'''
        
        plugin_registry_py = f'''"""
Plugin registry integration for generated backend.
"""

import logging
from typing import List, Optional
from namel3ss.plugins.registry import PluginRegistry, PluginSearchResult

logger = logging.getLogger(__name__)


class GeneratedBackendRegistry(PluginRegistry):
    """Plugin registry for this generated backend."""
    
    def __init__(self):
        super().__init__()
        self.required_plugins = {self._get_required_plugin_names(ir)}
        
    async def ensure_required_plugins(self) -> None:
        """Ensure all required plugins are available."""
        for plugin_name in self.required_plugins:
            try:
                # Check if plugin is available
                search_results = await self.search_plugins(plugin_name)
                
                if not search_results:
                    logger.error(f"Required plugin not found: {{plugin_name}}")
                    raise RuntimeError(f"Required plugin not available: {{plugin_name}}")
                    
                logger.info(f"Required plugin available: {{plugin_name}}")
                
            except Exception as e:
                logger.error(f"Error checking plugin {{plugin_name}}: {{e}}")
                raise
                
    async def install_missing_plugins(self) -> None:
        """Install any missing required plugins."""
        for plugin_name in self.required_plugins:
            try:
                # This would implement actual installation logic
                logger.info(f"Checking installation for: {{plugin_name}}")
                
            except Exception as e:
                logger.error(f"Error installing plugin {{plugin_name}}: {{e}}")
                raise


# Global registry instance
registry = GeneratedBackendRegistry()
'''
        
        return {
            "plugin_config.py": plugin_config_py,
            "plugin_registry.py": plugin_registry_py,
        }
        
    def _generate_plugin_aware_endpoints(self, ir: BackendIRWithPlugins, target: str) -> Dict[str, str]:
        """Generate endpoint implementations that use plugins."""
        
        endpoints_py = f'''"""
Generated endpoints with plugin support.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .plugin_config import get_plugin_config
from namel3ss.runtime.security import SecurityContext, get_security_context

router = APIRouter()

{self._generate_endpoint_handlers(ir)}
'''
        
        return {"endpoints.py": endpoints_py}
        
    # Helper methods for code generation
    
    def _generate_plugin_initialization(
        self,
        plugins: List[PluginSpec],
        optional: bool = False,
        indent: str = "    ",
    ) -> str:
        """Generate plugin initialization code."""
        if not plugins:
            return "" if not self.emit_comments else f"{indent}# No {'optional' if optional else 'required'} plugins"
            
        lines = []
        error_handling = "logger.warning" if optional else "logger.error"
        
        for plugin in plugins:
            lines.append(f"{indent}try:")
            lines.append(f"{indent}    plugin_instance = plugin_manager.load_plugin(")
            lines.append(f'{indent}        "{plugin.name}",')
            if plugin.version_constraint:
                lines.append(f'{indent}        version_constraint="{plugin.version_constraint}",')
            lines.append(f"{indent}    )")
            lines.append(f'{indent}    resolved_plugins["{plugin.name}"] = plugin_instance')
            lines.append(f'{indent}    logger.info(f"Loaded plugin: {plugin.name}")')
            lines.append(f"{indent}except Exception as e:")
            lines.append(f'{indent}    {error_handling}(f"Failed to load plugin {plugin.name}: {{e}}")')
            if not optional:
                lines.append(f"{indent}    raise")
            lines.append("")
            
        return "\n".join(lines)
        
    def _generate_plugin_requirements(self, requirements: PluginRequirementSpec) -> str:
        """Generate pip requirements for plugins."""
        if not requirements.required_plugins and not requirements.optional_plugins:
            return ""

        if not self.emit_comments:
            names = [plugin.name for plugin in requirements.required_plugins + requirements.optional_plugins]
            return "\n".join(names)
        
        lines = ["# Plugin dependencies"]
        
        for plugin in requirements.required_plugins + requirements.optional_plugins:
            # This would map plugin names to pip packages
            # For now, just add plugin name as comment
            lines.append(f"# {plugin.name}")
            if plugin.version_constraint:
                lines.append(f"# Version constraint: {plugin.version_constraint}")
                
        return "\n".join(lines)
        
    def _generate_plugin_configs(self, ir: BackendIRWithPlugins) -> str:
        """Generate plugin configuration dictionary."""
        lines = []
        
        for plugin in ir.plugin_requirements.required_plugins:
            config = {}  # Extract from plugin_config fields
            lines.append(f'    "{plugin.name}": {config},')
            
        if lines:
            return "\n".join(lines)
        return "" if not self.emit_comments else "    # No plugin configurations"
        
    def _get_required_plugin_names(self, ir: BackendIRWithPlugins) -> str:
        """Get set literal of required plugin names."""
        names = [f'"{p.name}"' for p in ir.plugin_requirements.required_plugins]
        return "{" + ", ".join(names) + "}"
        
    def _generate_endpoint_implementations(self, ir: BackendIRWithPlugins) -> str:
        """Generate FastAPI endpoint implementations."""
        # This would iterate through ir.endpoints and generate FastAPI routes
        return "" if not self.emit_comments else "# Generated endpoints will be added here"
        
    def _generate_serverless_routing(self, ir: BackendIRWithPlugins) -> str:
        """Generate serverless routing logic."""
        return "" if not self.emit_comments else "    # Generated routing logic will be added here"
        
    def _generate_serverless_handlers(self, ir: BackendIRWithPlugins) -> str:
        """Generate serverless handler functions."""
        return "" if not self.emit_comments else "# Generated handlers will be added here"
        
    def _generate_endpoint_handlers(self, ir: BackendIRWithPlugins) -> str:
        """Generate endpoint handler functions with plugin integration."""
        return "" if not self.emit_comments else "# Generated endpoint handlers with plugin support"


def create_plugin_aware_codegen(plugin_manager: PluginManager, *, emit_comments: bool = False) -> PluginAwareCodegen:
    """Factory function for plugin-aware codegen."""
    return PluginAwareCodegen(plugin_manager, emit_comments=emit_comments)


__all__ = [
    "PluginAwareCodegen",
    "PluginResolutionError",
    "create_plugin_aware_codegen",
]
