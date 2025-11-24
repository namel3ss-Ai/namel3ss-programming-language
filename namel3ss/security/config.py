"""
Security configuration and environment profile management.

This module handles loading, parsing, and managing security configurations
from namel3ss.toml and environment variables. It supports:

- Environment profiles (development, staging, production, sandbox)
- Policy stacking (global → environment → agent → tool)
- Runtime environment selection via CLI or env vars
- Configuration validation and defaults

Example namel3ss.toml:
----------------------

    [security]
    default_environment = "development"
    audit_log_path = "logs/security_audit.log"
    fail_mode = "closed"
    
    [security.global_policy]
    rate_limit_requests_per_minute = 100
    tool_timeout_seconds = 30.0
    llm_timeout_seconds = 120.0
    max_tokens_per_request = 8000
    
    [security.environments.development]
    allowed_permission_levels = ["READ_ONLY", "READ_WRITE", "NETWORK", "FILESYSTEM"]
    enforce_rate_limits = false
    warn_on_policy_violations = true
    
    [security.environments.production]
    allowed_permission_levels = ["READ_ONLY", "NETWORK"]
    deny_filesystem_access = true
    enforce_rate_limits = true
    require_https = true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore # Fallback

from namel3ss.ast.security import (
    Environment,
    PermissionLevel,
    SecurityPolicy,
    EnvironmentProfile,
    AuditLogLevel,
    InvalidSecurityConfig,
)


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class SecurityConfig:
    """
    Main security configuration for a Namel3ss workspace.
    
    Manages environment profiles, policies, and runtime security settings.
    Supports loading from namel3ss.toml and overriding via environment variables.
    """
    
    # Environment selection
    current_environment: Environment = Environment.DEVELOPMENT
    default_environment: Environment = Environment.DEVELOPMENT
    
    # Global settings
    audit_log_path: Optional[Path] = None
    audit_log_level: AuditLogLevel = AuditLogLevel.INFO
    fail_mode: str = "closed"  # closed = deny on error, open = allow on error
    
    # Global policy (applied to all unless overridden)
    global_policy: Optional[SecurityPolicy] = None
    
    # Environment profiles
    environments: Dict[Environment, EnvironmentProfile] = field(default_factory=dict)
    
    # Named policies (can be referenced by agents/tools)
    policies: Dict[str, SecurityPolicy] = field(default_factory=dict)
    
    # Runtime state
    audit_enabled: bool = True
    strict_mode: bool = False  # Fail on warnings
    
    def get_current_profile(self) -> EnvironmentProfile:
        """Get the active environment profile."""
        if self.current_environment in self.environments:
            return self.environments[self.current_environment]
        
        # Return default profile if not configured
        return self._create_default_profile(self.current_environment)
    
    def get_effective_policy(
        self,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        policy_name: Optional[str] = None
    ) -> SecurityPolicy:
        """
        Get the effective security policy by stacking:
        1. Global policy
        2. Environment policy
        3. Named policy (if provided)
        4. Agent-specific overrides (future)
        5. Tool-specific overrides (future)
        """
        # Start with global policy or create default
        effective = self.global_policy or self._create_default_policy()
        
        # Layer environment policy
        profile = self.get_current_profile()
        if profile.default_policy:
            effective = self._merge_policies(effective, profile.default_policy)
        
        # Layer named policy if provided
        if policy_name and policy_name in self.policies:
            effective = self._merge_policies(effective, self.policies[policy_name])
        
        return effective
    
    def is_permission_allowed(self, permission: PermissionLevel) -> bool:
        """Check if permission level is allowed in current environment."""
        profile = self.get_current_profile()
        
        # Check denied list first
        if permission in profile.denied_permission_levels:
            return False
        
        # Check allowed list (if specified)
        if profile.allowed_permission_levels:
            return permission in profile.allowed_permission_levels
        
        # Default: allow if not explicitly denied
        return True
    
    def is_tool_category_allowed(self, category: str) -> bool:
        """Check if tool category is allowed in current environment."""
        profile = self.get_current_profile()
        
        # Check denied list first
        if category in profile.denied_tool_categories:
            return False
        
        # Check allowed list (if specified)
        if profile.allowed_tool_categories:
            return category in profile.allowed_tool_categories
        
        # Default: allow if not explicitly denied
        return True
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if specific tool is allowed in current environment."""
        profile = self.get_current_profile()
        
        # Check denied list first
        if tool_name in profile.denied_tools:
            return False
        
        # Check allowed list (if specified)
        if profile.allowed_tools:
            return tool_name in profile.allowed_tools
        
        # Default: allow if not explicitly denied
        return True
    
    def _create_default_policy(self) -> SecurityPolicy:
        """Create a sensible default policy."""
        return SecurityPolicy(
            name="default",
            rate_limit_requests_per_minute=60,
            tool_timeout_seconds=30.0,
            llm_timeout_seconds=120.0,
            max_tokens_per_request=8000,
            max_concurrent_tool_calls=10,
            max_concurrent_llm_calls=5,
            fail_mode=self.fail_mode,
        )
    
    def _create_default_profile(self, env: Environment) -> EnvironmentProfile:
        """Create a default environment profile."""
        if env == Environment.PRODUCTION:
            return EnvironmentProfile(
                name="production",
                environment=Environment.PRODUCTION,
                allowed_permission_levels=[
                    PermissionLevel.READ_ONLY,
                    PermissionLevel.NETWORK,
                ],
                deny_filesystem_access=True,
                enforce_rate_limits=True,
                enforce_strict_timeouts=True,
                require_https=True,
                audit_log_level=AuditLogLevel.INFO,
            )
        elif env == Environment.STAGING:
            return EnvironmentProfile(
                name="staging",
                environment=Environment.STAGING,
                allowed_permission_levels=[
                    PermissionLevel.READ_ONLY,
                    PermissionLevel.READ_WRITE,
                    PermissionLevel.NETWORK,
                ],
                enforce_rate_limits=True,
                require_https=True,
                audit_log_level=AuditLogLevel.INFO,
            )
        elif env == Environment.SANDBOX:
            return EnvironmentProfile(
                name="sandbox",
                environment=Environment.SANDBOX,
                allowed_permission_levels=[PermissionLevel.READ_ONLY],
                deny_filesystem_access=True,
                deny_network_access=True,
                deny_database_access=True,
                enforce_rate_limits=True,
                audit_log_level=AuditLogLevel.FULL,
            )
        else:  # DEVELOPMENT
            return EnvironmentProfile(
                name="development",
                environment=Environment.DEVELOPMENT,
                allowed_permission_levels=list(PermissionLevel),  # All allowed
                enforce_rate_limits=False,
                warn_on_elevated_permissions=True,
                warn_on_policy_violations=True,
                audit_log_level=AuditLogLevel.DEBUG,
            )
    
    def _merge_policies(
        self,
        base: SecurityPolicy,
        override: SecurityPolicy
    ) -> SecurityPolicy:
        """Merge two policies, with override taking precedence."""
        return SecurityPolicy(
            name=f"{base.name}_merged_{override.name}",
            
            # Rate limiting (override wins)
            rate_limit_requests_per_minute=(
                override.rate_limit_requests_per_minute
                if override.rate_limit_requests_per_minute is not None
                else base.rate_limit_requests_per_minute
            ),
            rate_limit_requests_per_hour=(
                override.rate_limit_requests_per_hour
                if override.rate_limit_requests_per_hour is not None
                else base.rate_limit_requests_per_hour
            ),
            rate_limit_scope=override.rate_limit_scope or base.rate_limit_scope,
            
            # Timeouts (use minimum for safety)
            tool_timeout_seconds=min(
                override.tool_timeout_seconds,
                base.tool_timeout_seconds
            ),
            llm_timeout_seconds=min(
                override.llm_timeout_seconds,
                base.llm_timeout_seconds
            ),
            total_execution_timeout_seconds=(
                override.total_execution_timeout_seconds
                if override.total_execution_timeout_seconds is not None
                else base.total_execution_timeout_seconds
            ),
            
            # Token limits (use minimum for safety)
            max_tokens_per_request=(
                min(override.max_tokens_per_request, base.max_tokens_per_request)
                if override.max_tokens_per_request and base.max_tokens_per_request
                else override.max_tokens_per_request or base.max_tokens_per_request
            ),
            max_tokens_per_agent=(
                min(override.max_tokens_per_agent, base.max_tokens_per_agent)
                if override.max_tokens_per_agent and base.max_tokens_per_agent
                else override.max_tokens_per_agent or base.max_tokens_per_agent
            ),
            max_total_tokens=(
                min(override.max_total_tokens, base.max_total_tokens)
                if override.max_total_tokens and base.max_total_tokens
                else override.max_total_tokens or base.max_total_tokens
            ),
            
            # Cost controls (use minimum for safety)
            max_cost_per_request=(
                min(override.max_cost_per_request, base.max_cost_per_request)
                if override.max_cost_per_request and base.max_cost_per_request
                else override.max_cost_per_request or base.max_cost_per_request
            ),
            max_cost_per_agent=(
                min(override.max_cost_per_agent, base.max_cost_per_agent)
                if override.max_cost_per_agent and base.max_cost_per_agent
                else override.max_cost_per_agent or base.max_cost_per_agent
            ),
            max_total_cost=(
                min(override.max_total_cost, base.max_total_cost)
                if override.max_total_cost and base.max_total_cost
                else override.max_total_cost or base.max_total_cost
            ),
            
            # Concurrency (use minimum for safety)
            max_concurrent_tool_calls=min(
                override.max_concurrent_tool_calls,
                base.max_concurrent_tool_calls
            ),
            max_concurrent_llm_calls=min(
                override.max_concurrent_llm_calls,
                base.max_concurrent_llm_calls
            ),
            max_concurrent_agents=(
                min(override.max_concurrent_agents, base.max_concurrent_agents)
                if override.max_concurrent_agents and base.max_concurrent_agents
                else override.max_concurrent_agents or base.max_concurrent_agents
            ),
            
            # Strictness (most strict wins)
            enforce_strict=override.enforce_strict or base.enforce_strict,
            fail_mode=override.fail_mode if override.fail_mode != "closed" else base.fail_mode,
        )


# =============================================================================
# Configuration Loading
# =============================================================================


def load_security_config(config_path: Optional[Path] = None) -> SecurityConfig:
    """
    Load security configuration from namel3ss.toml.
    
    Args:
        config_path: Path to namel3ss.toml (default: search current/parent dirs)
    
    Returns:
        SecurityConfig instance
    
    Raises:
        InvalidSecurityConfig: If configuration is invalid
    """
    # Find config file
    if config_path is None:
        config_path = _find_config_file()
    
    if config_path is None or not config_path.exists():
        # No config file, use defaults
        return SecurityConfig()
    
    # Load TOML
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise InvalidSecurityConfig(f"Failed to parse {config_path}: {e}")
    
    security_data = data.get("security", {})
    
    # Parse global settings
    config = SecurityConfig()
    
    # Default environment
    default_env_str = security_data.get("default_environment", "development")
    try:
        config.default_environment = Environment(default_env_str.lower())
        config.current_environment = config.default_environment
    except ValueError:
        raise InvalidSecurityConfig(
            f"Invalid default_environment: {default_env_str}. "
            f"Must be one of: {[e.value for e in Environment]}"
        )
    
    # Override from environment variable
    env_override = os.getenv("NAMEL3SS_ENV") or os.getenv("NAMELESS_ENV")
    if env_override:
        try:
            config.current_environment = Environment(env_override.lower())
        except ValueError:
            raise InvalidSecurityConfig(
                f"Invalid NAMEL3SS_ENV: {env_override}. "
                f"Must be one of: {[e.value for e in Environment]}"
            )
    
    # Audit settings
    if "audit_log_path" in security_data:
        config.audit_log_path = Path(security_data["audit_log_path"])
    
    audit_level_str = security_data.get("audit_log_level", "info")
    try:
        config.audit_log_level = AuditLogLevel(audit_level_str.lower())
    except ValueError:
        raise InvalidSecurityConfig(f"Invalid audit_log_level: {audit_level_str}")
    
    config.fail_mode = security_data.get("fail_mode", "closed")
    config.audit_enabled = security_data.get("audit_enabled", True)
    config.strict_mode = security_data.get("strict_mode", False)
    
    # Load global policy
    if "global_policy" in security_data:
        config.global_policy = _parse_policy(
            "global",
            security_data["global_policy"],
            config.fail_mode
        )
    
    # Load environment profiles
    environments_data = security_data.get("environments", {})
    for env_name, env_data in environments_data.items():
        try:
            env = Environment(env_name.lower())
            profile = _parse_environment_profile(env_name, env, env_data)
            config.environments[env] = profile
        except ValueError:
            raise InvalidSecurityConfig(f"Invalid environment name: {env_name}")
    
    # Load named policies
    policies_data = security_data.get("policies", {})
    for policy_name, policy_data in policies_data.items():
        config.policies[policy_name] = _parse_policy(
            policy_name,
            policy_data,
            config.fail_mode
        )
    
    return config


def _find_config_file() -> Optional[Path]:
    """Search for namel3ss.toml in current and parent directories."""
    current = Path.cwd()
    for _ in range(10):  # Search up to 10 levels
        config_path = current / "namel3ss.toml"
        if config_path.exists():
            return config_path
        
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
    
    return None


def _parse_policy(name: str, data: Dict[str, Any], fail_mode: str) -> SecurityPolicy:
    """Parse a SecurityPolicy from TOML data."""
    return SecurityPolicy(
        name=name,
        rate_limit_requests_per_minute=data.get("rate_limit_requests_per_minute"),
        rate_limit_requests_per_hour=data.get("rate_limit_requests_per_hour"),
        rate_limit_scope=data.get("rate_limit_scope", "agent"),
        tool_timeout_seconds=data.get("tool_timeout_seconds", 30.0),
        llm_timeout_seconds=data.get("llm_timeout_seconds", 120.0),
        total_execution_timeout_seconds=data.get("total_execution_timeout_seconds"),
        max_tokens_per_request=data.get("max_tokens_per_request"),
        max_tokens_per_agent=data.get("max_tokens_per_agent"),
        max_total_tokens=data.get("max_total_tokens"),
        max_cost_per_request=data.get("max_cost_per_request"),
        max_cost_per_agent=data.get("max_cost_per_agent"),
        max_total_cost=data.get("max_total_cost"),
        max_concurrent_tool_calls=data.get("max_concurrent_tool_calls", 10),
        max_concurrent_llm_calls=data.get("max_concurrent_llm_calls", 5),
        max_concurrent_agents=data.get("max_concurrent_agents"),
        enforce_strict=data.get("enforce_strict", False),
        fail_mode=data.get("fail_mode", fail_mode),
    )


def _parse_environment_profile(
    name: str,
    env: Environment,
    data: Dict[str, Any]
) -> EnvironmentProfile:
    """Parse an EnvironmentProfile from TOML data."""
    # Parse permission levels
    allowed_permissions = []
    for perm_str in data.get("allowed_permission_levels", []):
        try:
            allowed_permissions.append(PermissionLevel(perm_str.lower()))
        except ValueError:
            raise InvalidSecurityConfig(
                f"Invalid permission level in {name}: {perm_str}"
            )
    
    denied_permissions = []
    for perm_str in data.get("denied_permission_levels", []):
        try:
            denied_permissions.append(PermissionLevel(perm_str.lower()))
        except ValueError:
            raise InvalidSecurityConfig(
                f"Invalid permission level in {name}: {perm_str}"
            )
    
    # Parse audit log level
    audit_level_str = data.get("audit_log_level", "info")
    try:
        audit_level = AuditLogLevel(audit_level_str.lower())
    except ValueError:
        raise InvalidSecurityConfig(
            f"Invalid audit_log_level in {name}: {audit_level_str}"
        )
    
    # Parse default policy if present
    default_policy = None
    if "default_policy" in data:
        default_policy = _parse_policy(
            f"{name}_default",
            data["default_policy"],
            data.get("fail_mode", "closed")
        )
    
    return EnvironmentProfile(
        name=name,
        environment=env,
        allowed_permission_levels=allowed_permissions,
        denied_permission_levels=denied_permissions,
        allowed_tool_categories=data.get("allowed_tool_categories", []),
        denied_tool_categories=data.get("denied_tool_categories", []),
        allowed_tools=data.get("allowed_tools", []),
        denied_tools=data.get("denied_tools", []),
        deny_filesystem_access=data.get("deny_filesystem_access", False),
        deny_network_access=data.get("deny_network_access", False),
        deny_database_access=data.get("deny_database_access", False),
        enforce_rate_limits=data.get("enforce_rate_limits", True),
        enforce_strict_timeouts=data.get("enforce_strict_timeouts", False),
        enforce_token_limits=data.get("enforce_token_limits", True),
        enforce_cost_limits=data.get("enforce_cost_limits", False),
        require_https=data.get("require_https", False),
        require_tls=data.get("require_tls", False),
        allowed_hosts=data.get("allowed_hosts", []),
        denied_hosts=data.get("denied_hosts", []),
        warn_on_elevated_permissions=data.get("warn_on_elevated_permissions", True),
        warn_on_policy_violations=data.get("warn_on_policy_violations", True),
        audit_log_level=audit_level,
        default_policy=default_policy,
    )


# =============================================================================
# Global Instance and Helpers
# =============================================================================


_global_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """
    Get the global security configuration.
    
    Loads configuration from namel3ss.toml on first call.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_security_config()
    return _global_config


def set_environment(env: Environment | str) -> None:
    """
    Set the current security environment.
    
    Args:
        env: Environment enum or string name
    """
    config = get_security_config()
    if isinstance(env, str):
        env = Environment(env.lower())
    config.current_environment = env


def reset_security_config() -> None:
    """Reset global security config (for testing)."""
    global _global_config
    _global_config = None


__all__ = [
    "SecurityConfig",
    "load_security_config",
    "get_security_config",
    "set_environment",
    "reset_security_config",
]
