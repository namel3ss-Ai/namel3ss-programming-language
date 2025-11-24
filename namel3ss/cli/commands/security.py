"""
Security command for the Namel3ss CLI.

Provides security validation, configuration inspection, and security reporting.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from namel3ss.parser import Parser
from namel3ss.security.config import get_security_config, load_security_config, set_environment
from namel3ss.security.validation import validate_application_security


def cmd_security(args: argparse.Namespace) -> int:
    """
    Main security command dispatcher.
    
    Dispatches to security subcommands (check, list-environments).
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if not hasattr(args, 'security_command') or args.security_command is None:
        print("usage: namel3ss security {check,list-environments} ...")
        print("\nSecurity validation and configuration management.")
        print("\nSubcommands:")
        print("  check              Validate application security configuration")
        print("  list-environments  List available security environments")
        return 1
    
    # Dispatch to appropriate subcommand via security_func
    if hasattr(args, 'security_func'):
        ctx = args.cli_context if hasattr(args, 'cli_context') else None
        return args.security_func(ctx, args)
    
    print(f"Unknown security command: {args.security_command}")
    return 1


def cmd_security_check(ctx, args: argparse.Namespace) -> int:
    """
    Validate security configuration and application security.
    
    Performs comprehensive security validation including:
    - Security configuration validation
    - Agent-tool access validation
    - Capability requirements validation
    - Permission level validation
    - Environment profile validation
    
    Args:
        ctx: CLI context
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Load security config
        if args.config_file:
            config_path = Path(args.config_file)
            if not config_path.exists():
                print(f"Error: Config file not found: {config_path}", file=sys.stderr)
                return 1
            load_security_config(config_path)
        
        # Set environment if specified
        if args.environment:
            set_environment(args.environment)
        
        security_config = get_security_config()
        
        # Display current security configuration
        if args.show_config:
            print("=" * 60)
            print("SECURITY CONFIGURATION")
            print("=" * 60)
            print(f"Environment: {security_config.default_environment}")
            print(f"Fail Mode: {security_config.fail_mode}")
            print(f"Audit Log: {security_config.audit_log_path or 'disabled'}")
            print(f"Audit Level: {security_config.audit_log_level}")
            print()
            
            # Show current environment profile
            env_profile = security_config.get_current_profile()
            if env_profile:
                print(f"Environment Profile: {env_profile.name}")
                # Convert PermissionLevel enums to strings
                allowed_levels = [str(level.value) if hasattr(level, 'value') else str(level) for level in env_profile.allowed_permission_levels]
                print(f"  Allowed Permission Levels: {', '.join(allowed_levels)}")
                print(f"  Denied Tools: {', '.join(env_profile.denied_tools) if env_profile.denied_tools else 'none'}")
                print(f"  Enforce Rate Limits: {env_profile.enforce_rate_limits}")
                print(f"  Enforce Strict Timeouts: {env_profile.enforce_strict_timeouts}")
                print(f"  Require HTTPS: {env_profile.require_https}")
                print(f"  Deny Filesystem Access: {env_profile.deny_filesystem_access}")
            print()
        
        # Validate application if file specified
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                return 1
            
            print(f"Validating security for: {file_path}")
            print()
            
            # Parse the application
            parser = Parser(file_path.read_text(), path=str(file_path))
            module = parser.parse()
            
            if not module.body:
                print("Error: No application found in file", file=sys.stderr)
                return 1
            
            app = module.body[0]
            
            # Validate security
            result = validate_application_security(app, security_config)
            
            # Display results
            if result.is_valid:
                print("✓ Security validation PASSED")
                print()
                print(f"Validated {len(app.agents)} agent(s) and {len(app.tools)} tool(s)")
                
                if result.warnings:
                    print()
                    print(f"Warnings ({len(result.warnings)}):")
                    for warning in result.warnings:
                        print(f"  ⚠ {warning}")
                
                return 0
            else:
                print("✗ Security validation FAILED")
                print()
                print(f"Errors ({len(result.errors)}):")
                for error in result.errors:
                    print(f"  ✗ {error}")
                
                if result.warnings:
                    print()
                    print(f"Warnings ({len(result.warnings)}):")
                    for warning in result.warnings:
                        print(f"  ⚠ {warning}")
                
                return 1
        
        # If just showing config, return success
        if args.show_config:
            return 0
        
        # No file specified and not showing config
        print("Use --file to validate an application or --show-config to view security settings")
        return 0
        
    except Exception as e:
        print(f"Error during security validation: {e}", file=sys.stderr)
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_security_list_environments(ctx, args: argparse.Namespace) -> int:
    """
    List available security environments and their configurations.
    
    Args:
        ctx: CLI context
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # Load security config
        if args.config_file:
            config_path = Path(args.config_file)
            if config_path.exists():
                load_security_config(config_path)
        
        security_config = get_security_config()
        
        print("=" * 60)
        print("AVAILABLE SECURITY ENVIRONMENTS")
        print("=" * 60)
        print()
        
        for env_name in ["development", "staging", "production", "sandbox"]:
            # Temporarily switch to environment to get its profile
            original_env = security_config.default_environment
            set_environment(env_name)
            env_profile = security_config.get_current_profile()
            set_environment(original_env.value if hasattr(original_env, 'value') else str(original_env))
            
            if env_profile:
                current = " (current)" if env_name == security_config.default_environment.value else ""
                print(f"📦 {env_name.upper()}{current}")
                # Convert PermissionLevel enums to strings
                allowed_levels = [str(level.value) if hasattr(level, 'value') else str(level) for level in env_profile.allowed_permission_levels]
                print(f"   Allowed Permissions: {', '.join(allowed_levels)}")
                print(f"   Rate Limits: {'enabled' if env_profile.enforce_rate_limits else 'disabled'}")
                print(f"   Strict Timeouts: {'enabled' if env_profile.enforce_strict_timeouts else 'disabled'}")
                if env_profile.denied_tools:
                    print(f"   Denied Tools: {', '.join(env_profile.denied_tools)}")
                print()
        
        return 0
        
    except Exception as e:
        print(f"Error listing environments: {e}", file=sys.stderr)
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def add_security_command(subparsers) -> None:
    """
    Add security command and subcommands to CLI.
    
    Args:
        subparsers: argparse subparsers object
    """
    # Main security command
    security_parser = subparsers.add_parser(
        'security',
        help='Security validation and configuration management'
    )
    
    security_subparsers = security_parser.add_subparsers(
        dest='security_command',
        help='Security subcommands'
    )
    
    # security check subcommand
    check_parser = security_subparsers.add_parser(
        'check',
        help='Validate application security configuration'
    )
    check_parser.add_argument(
        'file',
        nargs='?',
        help='Path to .ai/.n3 source file to validate'
    )
    check_parser.add_argument(
        '--config-file',
        help='Path to namel3ss.toml with security configuration'
    )
    check_parser.add_argument(
        '--environment', '--env',
        choices=['development', 'staging', 'production', 'sandbox'],
        help='Security environment to validate against'
    )
    check_parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display current security configuration'
    )
    check_parser.set_defaults(security_func=cmd_security_check)
    
    # security list-environments subcommand
    list_env_parser = security_subparsers.add_parser(
        'list-environments',
        aliases=['list-envs', 'envs'],
        help='List available security environments'
    )
    list_env_parser.add_argument(
        '--config-file',
        help='Path to namel3ss.toml with security configuration'
    )
    list_env_parser.set_defaults(security_func=cmd_security_list_environments)
    
    # Set main command dispatcher
    security_parser.set_defaults(func=cmd_security)


__all__ = [
    'cmd_security',
    'cmd_security_check',
    'cmd_security_list_environments',
    'add_security_command',
]
