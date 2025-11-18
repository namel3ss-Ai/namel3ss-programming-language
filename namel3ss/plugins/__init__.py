"""Plugin discovery and integration helpers for the Namel3ss CLI."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol


class CommandRegistrar(Protocol):
    """Protocol for registering custom CLI commands."""

    def add_parser(self, name: str, **kwargs):  # pragma: no cover - typing helper
        ...


class PluginHook(Protocol):
    """Protocol invoked around build and run lifecycles."""

    def __call__(self, *args, **kwargs) -> None:  # pragma: no cover - typing helper
        ...


@dataclass
class PluginCommand:
    """Command registered by a plugin."""

    name: str
    help: str
    handler: Callable[[object], None]
    configure: Optional[Callable[[object], None]] = None


@dataclass
class PluginRegistry:
    """Mutable registry exposed to plugin modules during discovery."""

    commands: List[PluginCommand] = field(default_factory=list)
    build_hooks: List[PluginHook] = field(default_factory=list)
    run_hooks: List[PluginHook] = field(default_factory=list)
    workspace_hooks: List[PluginHook] = field(default_factory=list)

    def add_command(
        self,
        name: str,
        handler: Callable[[object], None],
        *,
        help: str,
        configure: Optional[Callable[[object], None]] = None,
    ) -> None:
        self.commands.append(PluginCommand(name=name, help=help, handler=handler, configure=configure))

    def on_build(self, hook: PluginHook) -> None:
        self.build_hooks.append(hook)

    def on_run(self, hook: PluginHook) -> None:
        self.run_hooks.append(hook)

    def on_workspace(self, hook: PluginHook) -> None:
        self.workspace_hooks.append(hook)


class PluginManager:
    """Loads and wires plugins into the CLI."""

    def __init__(self, explicit_modules: Optional[Iterable[str]] = None) -> None:
        self.registry = PluginRegistry()
        self._explicit_modules = list(explicit_modules or [])
        self._loaded: Dict[str, object] = {}

    def load(self) -> None:
        for module_name in self._explicit_modules:
            self._load_module(module_name)

        for entry in self._iter_entry_points():
            try:
                plugin = entry.load()
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed to load plugin {entry.name}: {exc}") from exc
            self._register_plugin(entry.name, plugin)

    def _iter_entry_points(self):
        try:
            group = importlib.metadata.entry_points(group="namel3ss.plugins")
        except TypeError:  # pragma: no cover - Python <3.10 compatibility (not expected)
            group = importlib.metadata.entry_points().get("namel3ss.plugins", [])
        return group

    def _load_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        self._register_plugin(module_name, module)

    def _register_plugin(self, identifier: str, plugin: object) -> None:
        if identifier in self._loaded:
            return
        registrar = self.registry
        handler = None
        if hasattr(plugin, "register_plugin") and callable(plugin.register_plugin):
            handler = plugin.register_plugin
        elif hasattr(plugin, "register") and callable(plugin.register):
            handler = plugin.register
        elif callable(plugin):  # pragma: no cover - functional plugin style
            handler = plugin
        if handler is None:
            attr = getattr(plugin, "NAMEL3SS_PLUGIN", None)
            if attr is not None and callable(attr):
                handler = attr
        if handler is None:
            raise RuntimeError(
                f"Plugin '{identifier}' does not expose a register_plugin function or callable entry point."
            )
        handler(registrar)
        self._loaded[identifier] = plugin

    def register_commands(self, subparsers: CommandRegistrar) -> Dict[str, Callable[[object], None]]:
        handlers: Dict[str, Callable[[object], None]] = {}
        for cmd in self.registry.commands:
            parser = subparsers.add_parser(cmd.name, help=cmd.help)
            if cmd.configure:
                cmd.configure(parser)
            parser.set_defaults(func=cmd.handler)
            handlers[cmd.name] = cmd.handler
        return handlers

    def emit_build(self, *args, **kwargs) -> None:
        for hook in self.registry.build_hooks:
            hook(*args, **kwargs)

    def emit_run(self, *args, **kwargs) -> None:
        for hook in self.registry.run_hooks:
            hook(*args, **kwargs)

    def emit_workspace(self, *args, **kwargs) -> None:
        for hook in self.registry.workspace_hooks:
            hook(*args, **kwargs)


__all__ = [
    "CommandRegistrar",
    "PluginCommand",
    "PluginHook",
    "PluginManager",
    "PluginRegistry",
]

# Ensure built-in tool plugins are registered.
from . import builtins as _builtin_tool_plugins  # noqa: E402,F401
