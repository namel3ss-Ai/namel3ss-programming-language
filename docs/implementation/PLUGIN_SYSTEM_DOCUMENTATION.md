# Namel3ss Plugin System Documentation

## Overview

The Namel3ss Plugin System provides a production-grade, secure, and extensible plugin architecture for the Namel3ss AI programming language. It enables third-party developers to extend Namel3ss with custom connectors, tools, datasets, templates, and more.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Plugin Types](#plugin-types)
3. [Plugin Manifest](#plugin-manifest)
4. [Developing Plugins](#developing-plugins)
5. [Security Model](#security-model)
6. [Registry Integration](#registry-integration)
7. [CLI Reference](#cli-reference)
8. [API Reference](#api-reference)

## Quick Start

### Installing a Plugin

```bash
# Search for plugins
n3 plugin search "postgres connector"

# Install a plugin
n3 plugin install acme.postgres-connector

# List installed plugins
n3 plugin list

# Get plugin information
n3 plugin info acme.postgres-connector
```

### Using Plugins in N3 Code

```n3
// Import plugin
plugin postgres_connector from "acme.postgres-connector" version "^1.0.0"

// Use plugin-provided connector
connector db = postgres_connector.create({
    host: "localhost",
    database: "myapp",
    user: env.DB_USER,
    password: env.DB_PASSWORD
})

// Use in agent
agent DataAgent {
    tools: [db.query]
    
    async get_user(user_id: int) -> User {
        return await db.query("SELECT * FROM users WHERE id = ?", [user_id])
    }
}
```

### Creating a Plugin

```bash
# Create new plugin from template
n3 plugin create my-awesome-plugin \
  --author "Your Name" \
  --description "An awesome plugin" \
  --type connector

# Develop your plugin
cd my-awesome-plugin
# ... implement plugin logic ...

# Publish to registry
n3 plugin publish .
```

## Plugin Types

Namel3ss supports several plugin types:

### Connector Plugins
Provide connectivity to external systems (databases, APIs, message queues).

```python
# Example connector plugin
from namel3ss.plugins import ConnectorPlugin

class PostgresConnectorPlugin(ConnectorPlugin):
    async def create_connection(self, config):
        return PostgresConnection(config)
```

### Tool Plugins
Extend agents with custom tools and capabilities.

```python
# Example tool plugin
from namel3ss.plugins import ToolPlugin

class WebScrapingToolPlugin(ToolPlugin):
    async def scrape_url(self, url: str) -> dict:
        # Implementation
        return {"content": content, "metadata": metadata}
```

### Dataset Plugins
Provide access to custom data sources and formats.

```python
# Example dataset plugin
from namel3ss.plugins import DatasetPlugin

class CSVDatasetPlugin(DatasetPlugin):
    async def load_data(self, path: str) -> Iterator[dict]:
        # Implementation
        yield from csv_reader(path)
```

### Template Plugins
Provide reusable prompt templates, chains, or agent patterns.

```python
# Example template plugin
from namel3ss.plugins import TemplatePlugin

class ChatbotTemplatePlugin(TemplatePlugin):
    def get_templates(self):
        return {
            "customer_support": CustomerSupportTemplate(),
            "faq_bot": FAQBotTemplate(),
        }
```

### Provider Plugins
Integrate with new LLM providers or AI services.

```python
# Example provider plugin
from namel3ss.plugins import ProviderPlugin

class CustomLLMProviderPlugin(ProviderPlugin):
    async def generate(self, prompt: str, **kwargs) -> str:
        # Integration with custom LLM API
        return response
```

## Plugin Manifest

Every plugin must include a `n3-plugin.toml` manifest file:

```toml
# Basic plugin information
name = "my-awesome-plugin"
version = "1.0.0"
description = "An awesome plugin for Namel3ss"
license = "MIT"
homepage = "https://github.com/user/my-awesome-plugin"

# Author information
[author]
name = "Your Name"
email = "your.email@example.com"

# Plugin classification
plugin_type = "connector"  # or ["connector", "tool"] for multi-type

# Compatibility constraints
[compatibility]
namel3ss = ">=0.1.0"
python = ">=3.8"

# Entry points
[[entry_points]]
name = "main"
module = "my_awesome_plugin.main"

[[entry_points]]
name = "admin"
module = "my_awesome_plugin.admin"

# Security declarations
[security]
required_capabilities = ["network.http", "database.read"]
permission_level = "medium"  # low, medium, high
sandbox = true
network_access = ["api.example.com"]

# Dependencies
[dependencies]
requests = "^2.28.0"
sqlalchemy = "^1.4.0"

# Optional dependencies for specific features
[dependencies.optional]
redis = { version = "^4.0.0", feature = "caching" }

# Plugin-specific metadata
[metadata]
keywords = ["database", "postgresql", "connector"]
categories = ["databases", "connectors"]
```

### Manifest Fields Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique plugin identifier |
| `version` | string | Yes | Semantic version (e.g., "1.0.0") |
| `description` | string | Yes | Brief plugin description |
| `license` | string | No | License identifier (SPDX) |
| `homepage` | string | No | Plugin homepage/repository URL |
| `author.name` | string | Yes | Author name |
| `author.email` | string | No | Author email |
| `plugin_type` | string/array | Yes | Plugin type(s) |
| `compatibility.namel3ss` | string | Yes | Namel3ss version constraint |
| `entry_points` | array | Yes | Plugin entry points |
| `security.required_capabilities` | array | No | Required security capabilities |
| `security.permission_level` | string | No | Permission level (low/medium/high) |
| `dependencies` | object | No | Plugin dependencies |

## Developing Plugins

### Plugin Interface

All plugins implement the base `PluginInterface`:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class PluginInterface(ABC):
    """Base interface for all Namel3ss plugins."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod  
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
```

### Type-Specific Interfaces

```python
from namel3ss.plugins import (
    ConnectorPlugin,
    ToolPlugin, 
    DatasetPlugin,
    TemplatePlugin,
    ProviderPlugin,
)

# Connector example
class MyConnectorPlugin(ConnectorPlugin):
    async def initialize(self, config):
        self.connection_string = config["connection_string"]
    
    async def create_connection(self, config):
        return MyConnection(self.connection_string, config)
    
    async def cleanup(self):
        # Cleanup connections
        pass

# Tool example
class MyToolPlugin(ToolPlugin):
    async def initialize(self, config):
        self.api_key = config["api_key"]
    
    async def execute_tool(self, name: str, inputs: dict):
        if name == "search":
            return await self._search(inputs["query"])
        raise ValueError(f"Unknown tool: {name}")
    
    async def _search(self, query: str):
        # Implementation
        return {"results": [...]}
```

### Plugin Entry Points

The entry point module must provide a factory function:

```python
# my_plugin/main.py
from .plugin import MyAwesomePlugin

def create_plugin() -> MyAwesomePlugin:
    """Create plugin instance."""
    return MyAwesomePlugin()
```

### Testing Plugins

```python
# tests/test_my_plugin.py
import pytest
from my_plugin.main import create_plugin

@pytest.mark.asyncio
async def test_plugin_initialization():
    plugin = create_plugin()
    
    config = {"api_key": "test"}
    await plugin.initialize(config)
    
    assert plugin.api_key == "test"

@pytest.mark.asyncio 
async def test_plugin_functionality():
    plugin = create_plugin()
    await plugin.initialize({"api_key": "test"})
    
    result = await plugin.execute_tool("search", {"query": "test"})
    assert "results" in result
```

### Plugin Configuration

Plugins can define configuration schemas for validation:

```python
from pydantic import BaseModel
from namel3ss.plugins import PluginInterface

class MyPluginConfig(BaseModel):
    api_key: str
    timeout: float = 30.0
    retries: int = 3

class MyPlugin(PluginInterface):
    config_schema = MyPluginConfig
    
    async def initialize(self, config):
        # Config is automatically validated against schema
        self.config = self.config_schema(**config)
```

## Security Model

### Capability-Based Security

Plugins declare required capabilities in their manifest:

```toml
[security]
required_capabilities = [
    "network.http",          # HTTP network access
    "network.websocket",     # WebSocket access  
    "database.read",         # Database read access
    "database.write",        # Database write access
    "filesystem.read",       # File system read
    "filesystem.write",      # File system write
    "process.spawn",         # Process execution
    "env.read",             # Environment variable access
]
permission_level = "medium"
```

### Permission Levels

- **Low**: Basic functionality, no privileged operations
- **Medium**: Standard operations, limited system access
- **High**: Privileged operations, full system access

### Sandboxing

Plugins can be sandboxed for additional security:

```toml
[security]
sandbox = true
network_access = ["api.allowed-domain.com"]
filesystem_access = ["/tmp/plugin-data"]
```

### Code Signing

Production plugins should be signed:

```bash
# Sign plugin package
n3 plugin sign my-plugin.zip --key my-key.pem

# Verify signature  
n3 plugin verify my-plugin.zip --key public-key.pem
```

## Registry Integration

### Official Registry

The official Namel3ss Plugin Registry hosts verified, secure plugins.

### Registry Configuration

```bash
# Add custom registry
n3 registry add myregistry https://plugins.mycompany.com

# Set default registry
n3 registry default myregistry

# List registries
n3 registry list
```

### Publishing Plugins

```bash
# Login to registry
n3 registry login

# Publish plugin
n3 plugin publish . --registry official

# Update published plugin
n3 plugin publish . --version 1.0.1
```

### Registry API

Registries implement a standard REST API:

```
GET /api/v1/plugins/search?q=postgres
GET /api/v1/plugins/{name}
GET /api/v1/plugins/{name}/versions/{version}
POST /api/v1/plugins/{name}/versions/{version}
```

## CLI Reference

### Plugin Commands

```bash
# Search plugins
n3 plugin search [query] [options]
  --category TEXT      Filter by category
  --tag TEXT           Filter by tag  
  --type TEXT          Filter by plugin type
  --verified          Only verified plugins
  --min-rating FLOAT  Minimum rating
  --limit INTEGER     Max results
  --json              JSON output

# Install plugin
n3 plugin install NAME [options]
  --version TEXT      Specific version
  --registry TEXT     Source registry
  --force            Force reinstall
  --no-deps          Skip dependencies

# List installed plugins
n3 plugin list [options]
  --format TEXT       Output format (table/json/tree)
  --show-inactive     Include inactive plugins

# Plugin information
n3 plugin info NAME [options] 
  --version TEXT      Specific version
  --registry TEXT     Source registry
  --local            Local info only

# Uninstall plugin
n3 plugin uninstall NAME [options]
  --force            Skip confirmation

# Create new plugin
n3 plugin create NAME [options]
  --template TEXT     Plugin template
  --author TEXT       Author name
  --description TEXT  Description
  --type TEXT         Plugin type

# Publish plugin
n3 plugin publish PATH [options]
  --registry TEXT     Target registry
  --dry-run          Validate only
```

### Registry Commands

```bash
# Registry management
n3 registry add NAME URL
n3 registry remove NAME  
n3 registry list
n3 registry default NAME

# Authentication
n3 registry login [REGISTRY]
n3 registry logout [REGISTRY]
```

## API Reference

### Plugin Manager

```python
from namel3ss.plugins import PluginManager

manager = PluginManager()

# Discovery
manager.discover_plugins()
plugins = manager.get_all_plugins()

# Loading
plugin = manager.load_plugin("plugin-name")
manager.unload_plugin("plugin-name")

# Status
is_loaded = manager.is_plugin_loaded("plugin-name")
```

### Registry Client

```python
from namel3ss.plugins.registry_client import RegistryClient

client = RegistryClient()

# Search
results = await client.search_plugins(query="postgres")

# Download and install
plugin_path = await client.download_plugin("plugin-name")
install_path = await client.install_plugin("plugin-name")
```

### Plugin Manifest

```python
from namel3ss.plugins.manifest import PluginManifest, load_plugin_manifest

# Load from file
manifest = load_plugin_manifest("/path/to/plugin")

# Create programmatically
manifest = PluginManifest(
    name="my-plugin",
    version="1.0.0",
    description="My plugin",
    plugin_type="tool"
)
```

## Best Practices

### Plugin Development

1. **Follow semantic versioning** for plugin releases
2. **Declare minimal capabilities** needed for security
3. **Provide comprehensive documentation** and examples
4. **Include thorough testing** with good coverage
5. **Handle errors gracefully** with proper logging
6. **Support configuration validation** with schemas
7. **Implement proper cleanup** in the cleanup method

### Security

1. **Principle of least privilege** - request minimal capabilities
2. **Validate all inputs** from configuration and runtime
3. **Use secure communication** (HTTPS, TLS) for external APIs
4. **Don't store sensitive data** in plugin code or logs
5. **Follow secure coding practices** to prevent vulnerabilities

### Performance

1. **Lazy initialization** - defer expensive operations
2. **Connection pooling** for database and API connections  
3. **Caching** for frequently accessed data
4. **Async/await** for non-blocking operations
5. **Resource cleanup** to prevent memory leaks

### Compatibility

1. **Version constraints** - specify compatible Namel3ss versions
2. **Backward compatibility** - maintain API stability
3. **Deprecation notices** - provide migration paths
4. **Testing matrix** - test against multiple Namel3ss versions

## Examples

See the `examples/plugins/` directory for complete plugin examples:

- [PostgreSQL Connector](examples/plugins/postgres-connector/)
- [REST API Tool](examples/plugins/rest-api-tool/)
- [CSV Dataset](examples/plugins/csv-dataset/)
- [Chatbot Template](examples/plugins/chatbot-template/)
- [OpenAI Provider](examples/plugins/openai-provider/)

## Troubleshooting

### Common Issues

1. **Plugin not found**: Check plugin is installed and discoverable
2. **Permission denied**: Verify plugin has required capabilities
3. **Version conflicts**: Check compatibility constraints
4. **Import errors**: Ensure dependencies are installed
5. **Configuration errors**: Validate plugin configuration

### Debug Mode

```bash
# Enable debug logging
N3_LOG_LEVEL=DEBUG n3 plugin list

# Verbose plugin operations  
n3 plugin install my-plugin --verbose
```

### Support

- [GitHub Issues](https://github.com/namel3ss/namel3ss/issues)
- [Discord Community](https://discord.gg/namel3ss)
- [Plugin Registry](https://plugins.namel3ss.org)
- [Documentation](https://docs.namel3ss.org)