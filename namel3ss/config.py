"""Workspace configuration support for the Namel3ss CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = None  # type: ignore


@dataclass
class WorkspaceDefaults:
    """Default paths and flags applied to apps when not explicitly configured."""

    backend_out: Path = Path("backend_build")
    frontend_out: Path = Path("build")
    target: str = "static"
    host: str = "127.0.0.1"
    port: int = 8000
    frontend_port: Optional[int] = None
    enable_realtime: bool = False
    env: List[str] = field(default_factory=list)
    connector_retry_max_attempts: int = 3
    connector_retry_base_delay: float = 0.5
    connector_retry_max_delay: float = 5.0
    connector_concurrency_limit: int = 10
    # Symbolic expression safety limits
    expr_max_depth: int = 100
    expr_max_steps: int = 10000


@dataclass
class AppConfig:
    """Application-specific configuration resolved for the CLI."""

    name: str
    file: Path
    backend_out: Path
    frontend_out: Path
    port: int
    frontend_port: Optional[int] = None
    target: str = "static"
    enable_realtime: Optional[bool] = None
    env: List[str] = field(default_factory=list)
    connector_retry_max_attempts: Optional[int] = None
    connector_retry_base_delay: Optional[float] = None
    connector_retry_max_delay: Optional[float] = None
    connector_concurrency_limit: Optional[int] = None
    # Symbolic expression safety limits
    expr_max_depth: Optional[int] = None
    expr_max_steps: Optional[int] = None

    def merged_env(self, inherited: Sequence[str]) -> List[str]:
        if not inherited:
            return list(self.env)
        combined = list(inherited)
        for item in self.env:
            if item not in combined:
                combined.append(item)
        return combined


@dataclass
class ToolsConfig:
    """Configured integration commands."""

    test: Optional[str] = None
    lint: Optional[str] = None
    typecheck: Optional[str] = None
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkspaceConfig:
    """Resolved workspace configuration."""

    root: Path
    defaults: WorkspaceDefaults
    apps: Dict[str, AppConfig]
    plugins: List[str] = field(default_factory=list)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.apps

    def ensure_apps(self, candidates: Iterable[Path]) -> None:
        if self.apps:
            return
        auto_index = 0
        for app_path in candidates:
            if not app_path.exists() or app_path.suffix != ".n3":
                continue
            auto_index += 1
            name = app_path.stem
            backend_out = self.defaults.backend_out / name
            frontend_out = self.defaults.frontend_out / name
            self.apps[name] = AppConfig(
                name=name,
                file=app_path,
                backend_out=backend_out,
                frontend_out=frontend_out,
                port=self.defaults.port + auto_index - 1,
                frontend_port=(self.defaults.frontend_port + auto_index - 1)
                if self.defaults.frontend_port is not None
                else None,
                target=self.defaults.target,
                enable_realtime=self.defaults.enable_realtime,
                env=list(self.defaults.env),
                connector_retry_max_attempts=self.defaults.connector_retry_max_attempts,
                connector_retry_base_delay=self.defaults.connector_retry_base_delay,
                connector_retry_max_delay=self.defaults.connector_retry_max_delay,
                connector_concurrency_limit=self.defaults.connector_concurrency_limit,
            )

    def select(self, names: Optional[Sequence[str]]) -> List[AppConfig]:
        if not names:
            return list(self.apps.values())
        selected: List[AppConfig] = []
        for name in names:
            key = name if name in self.apps else None
            if key is None:
                for candidate, cfg in self.apps.items():
                    if cfg.file.name == name or cfg.file.as_posix() == name:
                        key = candidate
                        break
            if key is None:
                raise KeyError(f"App '{name}' is not defined in workspace config.")
            selected.append(self.apps[key])
        return selected


def _read_json_config(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def _read_toml_config(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("TOML parsing requires Python 3.11 or later.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _parse_apps(  # noqa: C901 - clarity preferred over deep factoring
    data: Dict[str, Any],
    defaults: WorkspaceDefaults,
    root: Path,
) -> Dict[str, AppConfig]:
    apps_section = data.get("apps") or {}
    apps: Dict[str, AppConfig] = {}
    for name, raw in apps_section.items():
        if not isinstance(raw, dict):
            continue
        file_path = Path(raw.get("file") or f"{name}.n3")
        if not file_path.is_absolute():
            file_path = root / file_path
        backend_raw = raw.get("backend_out") or defaults.backend_out
        backend_out = Path(backend_raw)
        if not backend_out.is_absolute():
            backend_out = (root / backend_out).resolve()
        frontend_raw = raw.get("frontend_out") or defaults.frontend_out
        frontend_out = Path(frontend_raw)
        if not frontend_out.is_absolute():
            frontend_out = (root / frontend_out).resolve()
        port = int(raw.get("port") or defaults.port)
        frontend_port = raw.get("frontend_port")
        if frontend_port is not None:
            frontend_port = int(frontend_port)
        target = str(raw.get("target") or defaults.target)
        enable_realtime = raw.get("enable_realtime")
        env_values = raw.get("env") or []
        env: List[str]
        if isinstance(env_values, (list, tuple)):
            env = [str(item) for item in env_values]
        elif isinstance(env_values, str):
            env = [env_values]
        else:
            env = []
        
        # Parse per-app connector overrides
        retry_max_attempts = raw.get("connector_retry_max_attempts")
        retry_base_delay = raw.get("connector_retry_base_delay")
        retry_max_delay = raw.get("connector_retry_max_delay")
        concurrency_limit = raw.get("connector_concurrency_limit")
        
        apps[name] = AppConfig(
            name=name,
            file=file_path,
            backend_out=backend_out,
            frontend_out=frontend_out,
            port=port,
            frontend_port=frontend_port,
            target=target,
            enable_realtime=enable_realtime,
            env=env,
            connector_retry_max_attempts=int(retry_max_attempts) if retry_max_attempts is not None else None,
            connector_retry_base_delay=float(retry_base_delay) if retry_base_delay is not None else None,
            connector_retry_max_delay=float(retry_max_delay) if retry_max_delay is not None else None,
            connector_concurrency_limit=int(concurrency_limit) if concurrency_limit is not None else None,
        )
    return apps


def _parse_defaults(data: Dict[str, Any], root: Path) -> WorkspaceDefaults:
    defaults_section = data.get("defaults") or {}
    backend_out = Path(defaults_section.get("backend_out") or WorkspaceDefaults.backend_out)
    if not backend_out.is_absolute():
        backend_out = (root / backend_out).resolve()
    frontend_out = Path(defaults_section.get("frontend_out") or WorkspaceDefaults.frontend_out)
    if not frontend_out.is_absolute():
        frontend_out = (root / frontend_out).resolve()
    target = str(defaults_section.get("target") or WorkspaceDefaults.target)
    host = str(defaults_section.get("host") or WorkspaceDefaults.host)
    port = int(defaults_section.get("port") or WorkspaceDefaults.port)
    frontend_port_raw = defaults_section.get("frontend_port")
    frontend_port = int(frontend_port_raw) if frontend_port_raw is not None else None
    enable_realtime = bool(defaults_section.get("enable_realtime", WorkspaceDefaults.enable_realtime))
    env_values = defaults_section.get("env") or []
    env: List[str]
    if isinstance(env_values, (list, tuple)):
        env = [str(item) for item in env_values]
    elif isinstance(env_values, str):
        env = [env_values]
    else:
        env = []
    
    # Parse connector retry/concurrency settings
    retry_max_attempts = int(defaults_section.get("connector_retry_max_attempts", WorkspaceDefaults.connector_retry_max_attempts))
    retry_base_delay = float(defaults_section.get("connector_retry_base_delay", WorkspaceDefaults.connector_retry_base_delay))
    retry_max_delay = float(defaults_section.get("connector_retry_max_delay", WorkspaceDefaults.connector_retry_max_delay))
    concurrency_limit = int(defaults_section.get("connector_concurrency_limit", WorkspaceDefaults.connector_concurrency_limit))
    
    return WorkspaceDefaults(
        backend_out=backend_out,
        frontend_out=frontend_out,
        target=target,
        host=host,
        port=port,
        frontend_port=frontend_port,
        enable_realtime=enable_realtime,
        env=env,
        connector_retry_max_attempts=retry_max_attempts,
        connector_retry_base_delay=retry_base_delay,
        connector_retry_max_delay=retry_max_delay,
        connector_concurrency_limit=concurrency_limit,
    )


def _parse_tools(data: Dict[str, Any]) -> ToolsConfig:
    tools_section = data.get("tools") or {}
    extras = {
        key: str(value)
        for key, value in tools_section.items()
        if key not in {"test", "lint", "typecheck"}
    }
    return ToolsConfig(
        test=str(tools_section.get("test")) if tools_section.get("test") else None,
        lint=str(tools_section.get("lint")) if tools_section.get("lint") else None,
        typecheck=str(tools_section.get("typecheck")) if tools_section.get("typecheck") else None,
        extras=extras,
    )


def locate_config_file(root: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = ["namel3ss.toml", ".namel3ssrc"]
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def load_workspace_config(root: Path, explicit: Optional[Path] = None) -> WorkspaceConfig:
    root = root.resolve()
    config_path = locate_config_file(root, explicit)
    if config_path is None:
        return WorkspaceConfig(root=root, defaults=WorkspaceDefaults(), apps={})

    if config_path.suffix == ".toml":
        data = _read_toml_config(config_path)
    else:
        data = _read_json_config(config_path)

    defaults = _parse_defaults(data, root)
    apps = _parse_apps(data, defaults, root)
    plugins_section = data.get("plugins") or []
    plugins: List[str]
    if isinstance(plugins_section, (list, tuple)):
        plugins = [str(entry) for entry in plugins_section]
    elif isinstance(plugins_section, str):
        plugins = [plugins_section]
    else:
        plugins = []

    tools = _parse_tools(data)

    return WorkspaceConfig(
        root=root,
        defaults=defaults,
        apps=apps,
        plugins=plugins,
        tools=tools,
        raw=data,
    )


def merge_env(base: Sequence[str], overrides: Sequence[str]) -> List[str]:
    result = list(base)
    for entry in overrides:
        if entry not in result:
            result.append(entry)
    return result


def default_app_name(path: Path) -> str:
    stem = path.stem
    return stem if stem else f"app_{abs(hash(path)) & 0xffff:x}"


def discover_n3_files(root: Path) -> List[Path]:
    return sorted(root.glob("*.n3"))


def resolve_apps_from_args(
    workspace: WorkspaceConfig,
    app_args: Optional[Sequence[str]],
    workspace_path: Path,
    *,
    discover: bool = True,
) -> List[AppConfig]:
    if discover:
        workspace.ensure_apps(discover_n3_files(workspace_path))
    if workspace.is_empty():
        raise FileNotFoundError("No applications defined. Provide --apps or create a namel3ss.toml configuration.")
    return workspace.select(app_args)


def apply_cli_overrides(
    apps: Iterable[AppConfig],
    *,
    backend_out: Optional[str] = None,
    frontend_out: Optional[str] = None,
    port: Optional[int] = None,
    frontend_port: Optional[int] = None,
    target: Optional[str] = None,
    enable_realtime: Optional[bool] = None,
) -> List[AppConfig]:
    updated: List[AppConfig] = []
    for index, app in enumerate(apps):
        backend = Path(backend_out) if backend_out else app.backend_out
        frontend = Path(frontend_out) if frontend_out else app.frontend_out
        if not backend.is_absolute():
            backend = (app.file.parent / backend).resolve()
        if not frontend.is_absolute():
            frontend = (app.file.parent / frontend).resolve()
        assigned_port = port + index if port is not None else app.port
        assigned_frontend_port = (
            (frontend_port + index if frontend_port is not None else app.frontend_port)
            if frontend_port is not None or app.frontend_port is not None
            else None
        )
        updated.append(
            AppConfig(
                name=app.name,
                file=app.file,
                backend_out=backend,
                frontend_out=frontend,
                port=assigned_port,
                frontend_port=assigned_frontend_port,
                target=target or app.target,
                enable_realtime=app.enable_realtime if enable_realtime is None else enable_realtime,
                env=list(app.env),
            )
        )
    return updated


def env_strings_from_config(entries: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for entry in entries:
        value = os.path.expandvars(entry)
        resolved.append(value)
    return resolved


def extract_connector_config(app_cfg: AppConfig, defaults: WorkspaceDefaults) -> Dict[str, Any]:
    """Extract connector runtime configuration from AppConfig with defaults fallback."""
    return {
        "retry_max_attempts": app_cfg.connector_retry_max_attempts if app_cfg.connector_retry_max_attempts is not None else defaults.connector_retry_max_attempts,
        "retry_base_delay": app_cfg.connector_retry_base_delay if app_cfg.connector_retry_base_delay is not None else defaults.connector_retry_base_delay,
        "retry_max_delay": app_cfg.connector_retry_max_delay if app_cfg.connector_retry_max_delay is not None else defaults.connector_retry_max_delay,
        "concurrency_limit": app_cfg.connector_concurrency_limit if app_cfg.connector_concurrency_limit is not None else defaults.connector_concurrency_limit,
    }
