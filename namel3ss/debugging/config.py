"""
Configuration management for namel3ss debugging system.

Provides configuration loading, validation, and environment integration
for debug settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

from namel3ss.debugging import DebugConfiguration, TraceFilter, TraceEventType


@dataclass
class DebugWorkspaceConfig:
    """
    Workspace-level debug configuration.
    
    Can be stored in .namel3ss/debug.json or environment variables.
    """
    
    # Global debug settings
    enabled: bool = False
    auto_trace: bool = False  # Automatically trace all executions
    
    # Output configuration
    trace_output_dir: str = "./debug/traces"
    trace_retention_days: int = 30
    
    # Default filtering
    default_components: List[str] = field(default_factory=list)  # Empty = all components
    default_event_types: List[str] = field(default_factory=list)  # Empty = all events
    
    # Performance settings
    max_trace_file_size_mb: int = 100
    max_events_per_file: int = 10000
    buffer_size: int = 1000
    flush_interval_seconds: float = 5.0
    
    # Advanced settings
    capture_memory: bool = True
    capture_performance: bool = True
    capture_stack_traces: bool = False  # For errors
    max_payload_size: int = 16 * 1024  # 16KB
    
    # Environment integration
    environment_override: bool = True  # Allow env vars to override config
    
    def to_debug_configuration(self) -> DebugConfiguration:
        """Convert to runtime DebugConfiguration."""
        
        # Apply environment overrides if enabled
        final_config = self
        if self.environment_override:
            final_config = self._apply_environment_overrides()
        
        # Create trace filter
        trace_filter = None
        if final_config.default_components or final_config.default_event_types:
            components = set(final_config.default_components) if final_config.default_components else None
            event_types = None
            if final_config.default_event_types:
                event_types = set(TraceEventType(t) for t in final_config.default_event_types)
            
            trace_filter = TraceFilter(
                components=components,
                event_types=event_types,
                include_performance=final_config.capture_performance,
                include_memory=final_config.capture_memory,
            )
        
        return DebugConfiguration(
            enabled=final_config.enabled,
            trace_output_dir=Path(final_config.trace_output_dir),
            buffer_events=True,
            buffer_size=final_config.buffer_size,
            flush_interval_seconds=final_config.flush_interval_seconds,
            trace_filter=trace_filter,
            max_event_payload_size=final_config.max_payload_size,
            capture_memory_usage=final_config.capture_memory,
            capture_performance_markers=final_config.capture_performance,
            trace_agent_execution=True,
            trace_prompt_execution=True,
            trace_chain_execution=True,
            trace_tool_calls=True,
            trace_llm_calls=True,
            trace_memory_operations=final_config.capture_memory,
        )
    
    def _apply_environment_overrides(self) -> "DebugWorkspaceConfig":
        """Apply environment variable overrides to configuration."""
        import copy
        
        config = copy.deepcopy(self)
        
        # Basic settings
        if os.getenv("NAMEL3SS_DEBUG_ENABLED"):
            config.enabled = os.getenv("NAMEL3SS_DEBUG_ENABLED", "false").lower() in ("true", "1", "yes")
        
        if os.getenv("NAMEL3SS_DEBUG_AUTO_TRACE"):
            config.auto_trace = os.getenv("NAMEL3SS_DEBUG_AUTO_TRACE", "false").lower() in ("true", "1", "yes")
        
        # Output settings
        if os.getenv("NAMEL3SS_DEBUG_OUTPUT_DIR"):
            config.trace_output_dir = os.getenv("NAMEL3SS_DEBUG_OUTPUT_DIR")
        
        if os.getenv("NAMEL3SS_DEBUG_RETENTION_DAYS"):
            try:
                config.trace_retention_days = int(os.getenv("NAMEL3SS_DEBUG_RETENTION_DAYS"))
            except ValueError:
                pass
        
        # Component filtering
        if os.getenv("NAMEL3SS_DEBUG_COMPONENTS"):
            config.default_components = os.getenv("NAMEL3SS_DEBUG_COMPONENTS").split(",")
        
        if os.getenv("NAMEL3SS_DEBUG_EVENT_TYPES"):
            config.default_event_types = os.getenv("NAMEL3SS_DEBUG_EVENT_TYPES").split(",")
        
        # Performance settings
        if os.getenv("NAMEL3SS_DEBUG_MAX_FILE_SIZE_MB"):
            try:
                config.max_trace_file_size_mb = int(os.getenv("NAMEL3SS_DEBUG_MAX_FILE_SIZE_MB"))
            except ValueError:
                pass
        
        if os.getenv("NAMEL3SS_DEBUG_BUFFER_SIZE"):
            try:
                config.buffer_size = int(os.getenv("NAMEL3SS_DEBUG_BUFFER_SIZE"))
            except ValueError:
                pass
        
        if os.getenv("NAMEL3SS_DEBUG_FLUSH_INTERVAL"):
            try:
                config.flush_interval_seconds = float(os.getenv("NAMEL3SS_DEBUG_FLUSH_INTERVAL"))
            except ValueError:
                pass
        
        # Feature flags
        if os.getenv("NAMEL3SS_DEBUG_CAPTURE_MEMORY"):
            config.capture_memory = os.getenv("NAMEL3SS_DEBUG_CAPTURE_MEMORY", "true").lower() in ("true", "1", "yes")
        
        if os.getenv("NAMEL3SS_DEBUG_CAPTURE_PERFORMANCE"):
            config.capture_performance = os.getenv("NAMEL3SS_DEBUG_CAPTURE_PERFORMANCE", "true").lower() in ("true", "1", "yes")
        
        if os.getenv("NAMEL3SS_DEBUG_CAPTURE_STACK_TRACES"):
            config.capture_stack_traces = os.getenv("NAMEL3SS_DEBUG_CAPTURE_STACK_TRACES", "false").lower() in ("true", "1", "yes")
        
        if os.getenv("NAMEL3SS_DEBUG_MAX_PAYLOAD_SIZE"):
            try:
                config.max_payload_size = int(os.getenv("NAMEL3SS_DEBUG_MAX_PAYLOAD_SIZE"))
            except ValueError:
                pass
        
        return config
    
    @classmethod
    def load_from_workspace(cls, workspace_root: Path) -> "DebugWorkspaceConfig":
        """Load debug configuration from workspace."""
        import json
        
        config_file = workspace_root / ".namel3ss" / "debug.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                # Log warning and use defaults
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load debug config from {config_file}: {e}")
        
        return cls()
    
    def save_to_workspace(self, workspace_root: Path):
        """Save debug configuration to workspace."""
        import json
        
        config_dir = workspace_root / ".namel3ss"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "debug.json"
        
        try:
            with open(config_file, "w") as f:
                json.dump(self.__dict__, f, indent=2)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save debug config to {config_file}: {e}")


class DebugConfigManager:
    """
    Manages debug configuration across workspace, environment, and CLI.
    
    Provides a unified interface for accessing debug settings from
    multiple sources with proper precedence.
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self._workspace_config: Optional[DebugWorkspaceConfig] = None
    
    def get_workspace_config(self) -> DebugWorkspaceConfig:
        """Get workspace debug configuration, loading if needed."""
        if self._workspace_config is None:
            self._workspace_config = DebugWorkspaceConfig.load_from_workspace(self.workspace_root)
        return self._workspace_config
    
    def get_runtime_config(
        self,
        *,
        cli_overrides: Optional[Dict[str, any]] = None
    ) -> DebugConfiguration:
        """
        Get final runtime debug configuration.
        
        Combines workspace config, environment variables, and CLI overrides.
        
        Args:
            cli_overrides: CLI-provided configuration overrides
        
        Returns:
            Final DebugConfiguration for runtime use
        """
        workspace_config = self.get_workspace_config()
        runtime_config = workspace_config.to_debug_configuration()
        
        # Apply CLI overrides if provided
        if cli_overrides:
            for key, value in cli_overrides.items():
                if hasattr(runtime_config, key) and value is not None:
                    setattr(runtime_config, key, value)
        
        return runtime_config
    
    def is_debug_enabled(self) -> bool:
        """Quick check if debugging is enabled."""
        return self.get_runtime_config().enabled
    
    def get_trace_output_dir(self) -> Path:
        """Get the configured trace output directory."""
        return self.get_runtime_config().trace_output_dir
    
    def cleanup_old_traces(self):
        """Clean up old trace files based on retention policy."""
        import time
        from datetime import datetime, timedelta
        
        workspace_config = self.get_workspace_config()
        trace_dir = Path(workspace_config.trace_output_dir)
        
        if not trace_dir.exists():
            return
        
        retention_days = workspace_config.trace_retention_days
        if retention_days <= 0:
            return  # Retention disabled
        
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        deleted_count = 0
        total_size = 0
        
        for trace_file in trace_dir.glob("*.jsonl"):
            try:
                if trace_file.stat().st_mtime < cutoff_time:
                    file_size = trace_file.stat().st_size
                    trace_file.unlink()
                    deleted_count += 1
                    total_size += file_size
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to delete old trace file {trace_file}: {e}")
        
        if deleted_count > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Cleaned up {deleted_count} old trace files ({total_size / 1024 / 1024:.1f}MB)")
    
    def validate_config(self) -> List[str]:
        """
        Validate debug configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        runtime_config = self.get_runtime_config()
        
        # Check output directory
        try:
            output_dir = runtime_config.trace_output_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
        except Exception as e:
            issues.append(f"Cannot write to trace output directory {runtime_config.trace_output_dir}: {e}")
        
        # Check buffer size
        if runtime_config.buffer_size <= 0:
            issues.append("Buffer size must be positive")
        
        if runtime_config.buffer_size > 100000:
            issues.append("Buffer size is very large (>100k), may impact performance")
        
        # Check payload size
        if runtime_config.max_event_payload_size <= 0:
            issues.append("Max payload size must be positive")
        
        # Check flush interval
        if runtime_config.flush_interval_seconds <= 0:
            issues.append("Flush interval must be positive")
        
        return issues


def get_debug_config_manager(workspace_root: Optional[Path] = None) -> DebugConfigManager:
    """Get debug configuration manager for the workspace."""
    return DebugConfigManager(workspace_root)


__all__ = [
    "DebugWorkspaceConfig",
    "DebugConfigManager",
    "get_debug_config_manager",
]