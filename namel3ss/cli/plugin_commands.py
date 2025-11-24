"""
CLI commands for plugin management in Namel3ss.

Provides command-line interface for discovering, installing, managing,
and developing plugins. Integrates with the plugin registry ecosystem.

Commands:
    - plugin search: Search for plugins
    - plugin install: Install plugins
    - plugin list: List installed plugins
    - plugin info: Get plugin information
    - plugin uninstall: Remove plugins
    - plugin create: Create new plugin scaffold
    - plugin publish: Publish plugin to registry
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..plugins.manager import PluginManager
from ..plugins.registry_client import RegistryClient, get_default_registry_client, RegistrySearchFilter
from ..plugins.manifest import PluginManifest, PluginType
from ..error import NamelessError

console = Console()


@click.group(name="plugin")
@click.pass_context
def plugin_cli(ctx):
    """Plugin management commands for Namel3ss."""
    if ctx.obj is None:
        ctx.obj = {}
    
    # Initialize plugin manager and registry client
    ctx.obj["plugin_manager"] = PluginManager()
    ctx.obj["registry_client"] = get_default_registry_client()


@plugin_cli.command()
@click.argument("query", required=False)
@click.option("--category", multiple=True, help="Filter by category")
@click.option("--tag", multiple=True, help="Filter by tag")
@click.option("--type", "plugin_types", multiple=True, help="Filter by plugin type")
@click.option("--verified", is_flag=True, help="Only show verified plugins")
@click.option("--min-rating", type=float, help="Minimum rating filter")
@click.option("--limit", default=20, help="Maximum number of results")
@click.option("--json-output", is_flag=True, help="Output results as JSON")
@click.pass_context
def search(ctx, query, category, tag, plugin_types, verified, min_rating, limit, json_output):
    """Search for plugins in the registry."""
    
    async def _search():
        registry_client = ctx.obj["registry_client"]
        
        search_filter = RegistrySearchFilter(
            query=query,
            categories=list(category) if category else None,
            tags=list(tag) if tag else None,
            plugin_types=list(plugin_types) if plugin_types else None,
            verified_only=verified,
            min_rating=min_rating,
        )
        
        try:
            results = await registry_client.search_plugins(
                query=query,
                categories=list(category) if category else None,
                tags=list(tag) if tag else None,
                plugin_types=list(plugin_types) if plugin_types else None,
                verified_only=verified,
                min_rating=min_rating,
            )
            
            # Collect all plugins from all backends
            all_plugins = []
            for backend_name, result in results.items():
                for plugin in result.plugins:
                    plugin_info = {
                        "name": plugin.name,
                        "version": str(plugin.version),
                        "description": plugin.description,
                        "publisher": plugin.publisher,
                        "registry": backend_name,
                        "downloads": plugin.stats.download_count,
                        "rating": plugin.stats.rating,
                        "verified": plugin.verified,
                        "types": [t.value for t in plugin.manifest.plugin_type if isinstance(plugin.manifest.plugin_type, list)] or [plugin.manifest.plugin_type.value],
                        "tags": list(plugin.tags),
                    }
                    all_plugins.append((plugin, plugin_info))
            
            # Sort by downloads descending
            all_plugins.sort(key=lambda x: x[1]["downloads"], reverse=True)
            
            # Limit results
            all_plugins = all_plugins[:limit]
            
            if json_output:
                console.print_json(data=[p[1] for p in all_plugins])
                return
            
            if not all_plugins:
                console.print("[yellow]No plugins found matching your criteria.[/yellow]")
                return
            
            # Create rich table
            table = Table(title=f"Plugin Search Results ({len(all_plugins)} found)")
            table.add_column("Name", style="bold blue")
            table.add_column("Version", style="green")
            table.add_column("Description", style="white", max_width=40)
            table.add_column("Publisher", style="cyan")
            table.add_column("Downloads", justify="right")
            table.add_column("Rating", justify="right")
            table.add_column("Registry", style="magenta")
            
            for plugin, info in all_plugins:
                name_text = Text(info["name"])
                if info["verified"]:
                    name_text.append(" ✓", style="bold green")
                
                rating_text = f"{info['rating']:.1f}" if info["rating"] > 0 else "N/A"
                downloads_text = f"{info['downloads']:,}" if info["downloads"] > 0 else "0"
                
                table.add_row(
                    name_text,
                    info["version"],
                    info["description"][:40] + "..." if len(info["description"]) > 40 else info["description"],
                    info["publisher"],
                    downloads_text,
                    rating_text,
                    info["registry"],
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error searching plugins: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_search())


@plugin_cli.command()
@click.argument("name")
@click.option("--version", help="Specific version to install")
@click.option("--registry", help="Registry to install from")
@click.option("--force", is_flag=True, help="Force reinstall if already installed")
@click.option("--no-deps", is_flag=True, help="Don't install dependencies")
@click.pass_context
def install(ctx, name, version, registry, force, no_deps):
    """Install a plugin."""
    
    async def _install():
        plugin_manager = ctx.obj["plugin_manager"]
        registry_client = ctx.obj["registry_client"]
        
        try:
            # Check if already installed
            if not force:
                try:
                    existing = plugin_manager.get_plugin(name)
                    console.print(f"[yellow]Plugin {name} is already installed (version {existing.version})[/yellow]")
                    console.print("Use --force to reinstall")
                    return
                except Exception:
                    pass  # Not installed, proceed
            
            with console.status(f"[bold green]Installing {name}..."):
                # Download and install
                plugin_dir = await registry_client.install_plugin(
                    name=name,
                    version=version,
                    backend=registry,
                )
                
                # Load plugin to verify installation
                plugin_instance = plugin_manager.load_plugin(name)
                
                console.print(f"[green]✓[/green] Successfully installed {name} v{plugin_instance.version}")
                
                if not no_deps:
                    # TODO: Install dependencies
                    pass
                
        except Exception as e:
            console.print(f"[red]Error installing plugin {name}: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_install())


@plugin_cli.command("list")
@click.option("--format", type=click.Choice(["table", "json", "tree"]), default="table", help="Output format")
@click.option("--show-inactive", is_flag=True, help="Show inactive plugins")
@click.pass_context
def list_plugins(ctx, format, show_inactive):
    """List installed plugins."""
    
    plugin_manager = ctx.obj["plugin_manager"]
    
    try:
        # Discover all plugins
        plugin_manager.discover_plugins()
        plugins = plugin_manager.get_all_plugins()
        
        if not plugins:
            console.print("[yellow]No plugins installed.[/yellow]")
            return
        
        if format == "json":
            plugin_data = []
            for plugin in plugins:
                data = {
                    "name": plugin.name,
                    "version": str(plugin.version),
                    "description": plugin.description,
                    "types": [t.value for t in plugin.plugin_type] if isinstance(plugin.plugin_type, list) else [plugin.plugin_type.value],
                    "active": plugin_manager.is_plugin_loaded(plugin.name),
                    "author": plugin.author.name if plugin.author else None,
                    "entry_points": len(plugin.entry_points),
                }
                plugin_data.append(data)
            
            console.print_json(data=plugin_data)
            return
        
        elif format == "tree":
            tree = Tree("[bold blue]Installed Plugins[/bold blue]")
            
            for plugin in plugins:
                is_active = plugin_manager.is_plugin_loaded(plugin.name)
                status_icon = "🟢" if is_active else "⚪"
                
                if not is_active and not show_inactive:
                    continue
                
                plugin_node = tree.add(f"{status_icon} {plugin.name} v{plugin.version}")
                plugin_node.add(f"[dim]{plugin.description}[/dim]")
                
                if plugin.author:
                    plugin_node.add(f"[cyan]Author:[/cyan] {plugin.author.name}")
                
                types = plugin.plugin_type if isinstance(plugin.plugin_type, list) else [plugin.plugin_type]
                plugin_node.add(f"[magenta]Types:[/magenta] {', '.join(t.value for t in types)}")
                
                if plugin.entry_points:
                    entry_node = plugin_node.add("[yellow]Entry Points:[/yellow]")
                    for entry_point in plugin.entry_points:
                        entry_node.add(f"• {entry_point.name} → {entry_point.module}")
            
            console.print(tree)
            return
        
        # Default table format
        table = Table(title=f"Installed Plugins ({len(plugins)} total)")
        table.add_column("Name", style="bold blue")
        table.add_column("Version", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Type(s)", style="magenta")
        table.add_column("Description", max_width=40)
        table.add_column("Author", style="cyan")
        
        for plugin in plugins:
            is_active = plugin_manager.is_plugin_loaded(plugin.name)
            
            if not is_active and not show_inactive:
                continue
                
            status = "🟢 Active" if is_active else "⚪ Inactive"
            
            types = plugin.plugin_type if isinstance(plugin.plugin_type, list) else [plugin.plugin_type]
            type_str = ", ".join(t.value for t in types)
            
            description = plugin.description[:40] + "..." if len(plugin.description) > 40 else plugin.description
            author_name = plugin.author.name if plugin.author else "Unknown"
            
            table.add_row(
                plugin.name,
                str(plugin.version),
                status,
                type_str,
                description,
                author_name,
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing plugins: {e}[/red]")
        sys.exit(1)


@plugin_cli.command()
@click.argument("name")
@click.option("--version", help="Specific version to get info for")
@click.option("--registry", help="Registry to query")
@click.option("--local", is_flag=True, help="Show local plugin info only")
@click.pass_context
def info(ctx, name, version, registry, local):
    """Get detailed information about a plugin."""
    
    async def _info():
        plugin_manager = ctx.obj["plugin_manager"]
        registry_client = ctx.obj["registry_client"]
        
        try:
            if local:
                # Show local plugin info
                try:
                    plugin = plugin_manager.get_plugin(name)
                    _show_local_plugin_info(plugin)
                except Exception as e:
                    console.print(f"[red]Plugin {name} not found locally: {e}[/red]")
                    sys.exit(1)
            else:
                # Show registry info
                try:
                    plugin_metadata = await registry_client.get_plugin(name, version, registry)
                    _show_registry_plugin_info(plugin_metadata)
                except Exception as e:
                    console.print(f"[red]Error getting plugin info: {e}[/red]")
                    sys.exit(1)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    def _show_local_plugin_info(plugin):
        """Show information for locally installed plugin."""
        console.print(f"[bold blue]{plugin.name}[/bold blue] v{plugin.version}")
        console.print(f"[dim]{plugin.description}[/dim]")
        console.print()
        
        table = Table.grid(padding=1)
        table.add_column("Field", style="cyan", min_width=15)
        table.add_column("Value")
        
        table.add_row("Author", plugin.author.name if plugin.author else "Unknown")
        table.add_row("License", plugin.license or "Not specified")
        
        types = plugin.plugin_type if isinstance(plugin.plugin_type, list) else [plugin.plugin_type]
        table.add_row("Types", ", ".join(t.value for t in types))
        
        table.add_row("Entry Points", str(len(plugin.entry_points)))
        
        if plugin.dependencies:
            deps = ", ".join(f"{k}{v}" for k, v in plugin.dependencies.items())
            table.add_row("Dependencies", deps)
        
        if plugin.security:
            caps = ", ".join(plugin.security.required_capabilities)
            table.add_row("Capabilities", caps or "None")
        
        console.print(table)
        
        # Show entry points
        if plugin.entry_points:
            console.print("\n[bold yellow]Entry Points:[/bold yellow]")
            for entry_point in plugin.entry_points:
                console.print(f"  • {entry_point.name}: {entry_point.module}")
    
    def _show_registry_plugin_info(plugin_metadata):
        """Show information for plugin from registry."""
        console.print(f"[bold blue]{plugin_metadata.name}[/bold blue] v{plugin_metadata.version}")
        console.print(f"[dim]{plugin_metadata.description}[/dim]")
        
        if plugin_metadata.verified:
            console.print("[green]✓ Verified Publisher[/green]")
        
        console.print()
        
        table = Table.grid(padding=1)
        table.add_column("Field", style="cyan", min_width=15)
        table.add_column("Value")
        
        table.add_row("Publisher", plugin_metadata.publisher)
        table.add_row("Published", plugin_metadata.published_at.strftime("%Y-%m-%d"))
        table.add_row("Downloads", f"{plugin_metadata.stats.download_count:,}")
        
        if plugin_metadata.stats.rating > 0:
            table.add_row("Rating", f"{plugin_metadata.stats.rating:.1f}/5.0 ({plugin_metadata.stats.rating_count} reviews)")
        
        if plugin_metadata.tags:
            table.add_row("Tags", ", ".join(sorted(plugin_metadata.tags)))
        
        if plugin_metadata.categories:
            table.add_row("Categories", ", ".join(sorted(plugin_metadata.categories)))
        
        # URLs
        if plugin_metadata.homepage_url:
            table.add_row("Homepage", plugin_metadata.homepage_url)
        if plugin_metadata.documentation_url:
            table.add_row("Documentation", plugin_metadata.documentation_url)
        
        console.print(table)
    
    asyncio.run(_info())


@plugin_cli.command()
@click.argument("name")
@click.option("--force", is_flag=True, help="Force removal without confirmation")
@click.pass_context
def uninstall(ctx, name, force):
    """Uninstall a plugin."""
    
    plugin_manager = ctx.obj["plugin_manager"]
    
    try:
        # Check if plugin exists
        try:
            plugin = plugin_manager.get_plugin(name)
        except Exception:
            console.print(f"[red]Plugin {name} is not installed.[/red]")
            sys.exit(1)
        
        # Confirm removal
        if not force:
            confirmed = click.confirm(f"Remove plugin {name} v{plugin.version}?")
            if not confirmed:
                console.print("Cancelled.")
                return
        
        # Unload and remove
        if plugin_manager.is_plugin_loaded(name):
            plugin_manager.unload_plugin(name)
        
        # Remove plugin files (basic implementation)
        plugin_dir = Path.home() / ".namel3ss" / "plugins" / name
        if plugin_dir.exists():
            import shutil
            shutil.rmtree(plugin_dir)
            console.print(f"[green]✓[/green] Successfully removed {name}")
        else:
            console.print(f"[yellow]Plugin files not found for {name}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error removing plugin {name}: {e}[/red]")
        sys.exit(1)


@plugin_cli.command()
@click.argument("name")
@click.option("--template", default="basic", help="Plugin template to use")
@click.option("--author", prompt=True, help="Plugin author name")
@click.option("--description", prompt=True, help="Plugin description")
@click.option("--type", "plugin_type", 
              type=click.Choice(["connector", "tool", "dataset", "template", "provider", "evaluator", "transformer", "mixed"]),
              prompt=True, help="Plugin type")
@click.pass_context
def create(ctx, name, template, author, description, plugin_type):
    """Create a new plugin from template."""
    
    try:
        plugin_dir = Path.cwd() / name
        
        if plugin_dir.exists():
            console.print(f"[red]Directory {name} already exists.[/red]")
            sys.exit(1)
        
        # Create plugin structure
        plugin_dir.mkdir()
        
        # Create manifest
        manifest = {
            "name": name,
            "version": "0.1.0",
            "description": description,
            "author": {
                "name": author,
            },
            "license": "MIT",
            "plugin_type": plugin_type,
            "compatibility": {
                "namel3ss": ">=0.1.0",
            },
            "entry_points": [
                {
                    "name": "main",
                    "module": f"{name}.main",
                }
            ],
        }
        
        manifest_path = plugin_dir / "n3-plugin.toml"
        with manifest_path.open("w") as f:
            import toml
            toml.dump(manifest, f)
        
        # Create main module
        main_py = f'''"""
{name} - A Namel3ss plugin.

{description}
"""

from typing import Any, Dict
from namel3ss.plugins import PluginInterface

class {name.title().replace('_', '').replace('-', '')}Plugin(PluginInterface):
    """Main plugin class for {name}."""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config = config
        
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    # Add plugin-specific methods here


# Entry point function
def create_plugin() -> {name.title().replace('_', '').replace('-', '')}Plugin:
    """Create plugin instance."""
    return {name.title().replace('_', '').replace('-', '')}Plugin()
'''
        
        # Create module directory
        module_dir = plugin_dir / name.replace('-', '_')
        module_dir.mkdir()
        
        (module_dir / "__init__.py").write_text("")
        (module_dir / "main.py").write_text(main_py)
        
        # Create README
        readme_md = f'''# {name}

{description}

## Installation

```bash
n3 plugin install {name}
```

## Usage

[Add usage instructions here]

## Development

[Add development instructions here]

## License

MIT
'''
        
        (plugin_dir / "README.md").write_text(readme_md)
        
        # Create setup.py for Python packaging
        setup_py = f'''from setuptools import setup, find_packages

setup(
    name="{name}",
    version="0.1.0", 
    description="{description}",
    author="{author}",
    packages=find_packages(),
    install_requires=[
        "namel3ss>=0.1.0",
    ],
    entry_points={{
        "namel3ss.plugins": [
            "main = {name.replace('-', '_')}.main:create_plugin",
        ],
    }},
)
'''
        
        (plugin_dir / "setup.py").write_text(setup_py)
        
        console.print(f"[green]✓[/green] Created plugin {name} in {plugin_dir}")
        console.print("\nNext steps:")
        console.print(f"  cd {name}")
        console.print("  # Implement your plugin logic")
        console.print(f"  n3 plugin publish {name}")
        
    except Exception as e:
        console.print(f"[red]Error creating plugin: {e}[/red]")
        sys.exit(1)


@plugin_cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--registry", help="Registry to publish to")
@click.option("--dry-run", is_flag=True, help="Validate without publishing")
@click.pass_context
def publish(ctx, path, registry, dry_run):
    """Publish a plugin to a registry."""
    
    async def _publish():
        try:
            plugin_path = Path(path)
            
            # Load and validate manifest
            manifest_path = plugin_path / "n3-plugin.toml"
            if not manifest_path.exists():
                console.print(f"[red]No plugin manifest found at {manifest_path}[/red]")
                sys.exit(1)
            
            with manifest_path.open() as f:
                import toml
                manifest_data = toml.load(f)
                
            manifest = PluginManifest(**manifest_data)
            
            console.print(f"[green]✓[/green] Plugin manifest valid")
            console.print(f"Publishing {manifest.name} v{manifest.version}")
            
            if dry_run:
                console.print("[yellow]Dry run - would publish to registry[/yellow]")
                return
            
            # TODO: Implement actual publishing
            console.print("[yellow]Publishing not yet implemented[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error publishing plugin: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_publish())


# Register the plugin CLI with the main CLI
def register_plugin_commands(main_cli):
    """Register plugin commands with main CLI."""
    main_cli.add_command(plugin_cli)


if __name__ == "__main__":
    plugin_cli()