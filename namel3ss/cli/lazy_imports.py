"""
Lazy imports for CLI commands that require optional dependencies.

This module provides lazy loading for commands that depend on optional extras,
allowing the core CLI to work with minimal dependencies while providing
helpful error messages when optional features are used.
"""

import sys
from typing import Any, Optional

def lazy_import_command(module_path: str, command_name: str, extra_name: str, description: str):
    """
    Create a lazy-loaded command that shows a helpful error if dependencies are missing.
    
    Args:
        module_path: Path to the module containing the command
        command_name: Name of the command function to import
        extra_name: Name of the pip extra that provides the dependencies
        description: Human-readable description of what the command does
    """
    def lazy_command(*args, **kwargs):
        try:
            # Try to import the actual command
            module = __import__(module_path, fromlist=[command_name])
            command_func = getattr(module, command_name)
            return command_func(*args, **kwargs)
        except ImportError as e:
            # Show helpful error message
            print(f"Error: {description} requires the '{extra_name}' extra.")
            print(f"Install with: pip install 'namel3ss[{extra_name}]'")
            print(f"Or for all features: pip install 'namel3ss[all]'")
            print(f"\nMissing dependency: {e}")
            sys.exit(1)
    
    return lazy_command

def try_import_optional(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Try to import an optional module, returning None if it's not available.
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError:
        return None

# Core commands that work with minimal dependencies
CORE_COMMANDS = {
    'build': ('namel3ss.cli.commands.build', 'cmd_build'),
    'run': ('namel3ss.cli.commands.run', 'cmd_run'), 
    'doctor': ('namel3ss.cli.commands.doctor', 'cmd_doctor'),
}

# Commands that require the 'cli' extra (rich output, debugging, etc.)
CLI_COMMANDS = {
    'debug': ('namel3ss.cli.commands.debug', 'cmd_debug', 'cli', 'Advanced debugging features'),
    'conformance': ('namel3ss.cli.commands.conformance', 'cmd_conformance', 'cli', 'Language conformance testing'),
    'packages': ('namel3ss.cli.commands.packages', 'cmd_packages', 'cli', 'Package management operations'),
    'modules': ('namel3ss.cli.commands.modules', 'cmd_modules', 'cli', 'Module introspection'),
    'security': ('namel3ss.cli.commands.security', 'add_security_command', 'cli', 'Security validation'),
    'stdlib': ('namel3ss.cli.commands.cmd_stdlib', 'cmd_stdlib', 'cli', 'Standard library operations'),
}

# Commands that require local deployment capabilities
LOCAL_DEPLOY_COMMANDS = {
    'deploy-local': ('namel3ss.cli.commands.local_deploy', 'cmd_deploy_local', 'local-deploy', 'Local model deployment'),
}

# Commands that require AI/LLM features  
AI_COMMANDS = {
    'eval': ('namel3ss.cli.commands.eval', 'cmd_eval', 'ai', 'AI model evaluation'),
    'eval-suite': ('namel3ss.cli.commands.eval', 'cmd_eval_suite', 'ai', 'Comprehensive evaluation suite'),
    'train': ('namel3ss.cli.commands.train', 'cmd_train', 'ai', 'Model training and fine-tuning'),
}

# Commands that require development tools
DEV_COMMANDS = {
    'format': ('namel3ss.cli.commands.tools', 'cmd_format', 'dev', 'Code formatting'),
    'lint': ('namel3ss.cli.commands.tools', 'cmd_lint', 'dev', 'Code linting'),
    'typecheck': ('namel3ss.cli.commands.tools', 'cmd_typecheck', 'dev', 'Type checking'),
    'test': ('namel3ss.cli.commands.tools', 'cmd_test', 'dev', 'Testing framework'),
    'lsp': ('namel3ss.cli.commands.tools', 'cmd_lsp', 'dev', 'Language server protocol'),
}

def get_command(command_name: str):
    """
    Get a command function, either directly or as a lazy loader.
    """
    # Check core commands first (always available)
    if command_name in CORE_COMMANDS:
        module_path, func_name = CORE_COMMANDS[command_name]
        module = __import__(module_path, fromlist=[func_name])
        return getattr(module, func_name)
    
    # Check optional commands
    all_optional_commands = {
        **CLI_COMMANDS,
        **LOCAL_DEPLOY_COMMANDS, 
        **AI_COMMANDS,
        **DEV_COMMANDS
    }
    
    if command_name in all_optional_commands:
        module_path, func_name, extra, description = all_optional_commands[command_name]
        return lazy_import_command(module_path, func_name, extra, description)
    
    raise ValueError(f"Unknown command: {command_name}")

# Specific lazy functions for CLI command registration
def lazy_cmd_eval(args):
    """Lazy loader for eval command."""
    return lazy_import_command('namel3ss.cli.commands.eval', 'cmd_eval', 'ai', 'AI model evaluation')(args)

def lazy_cmd_eval_suite(args):
    """Lazy loader for eval-suite command."""
    return lazy_import_command('namel3ss.cli.commands.eval', 'cmd_eval_suite', 'ai', 'Comprehensive evaluation suite')(args)

def lazy_cmd_train(args):
    """Lazy loader for train command."""
    return lazy_import_command('namel3ss.cli.commands.train', 'cmd_train', 'rlhf', 'Model training and fine-tuning')(args)

def lazy_cmd_deploy(args):
    """Lazy loader for cloud deploy command."""
    return lazy_import_command('namel3ss.cli.commands.deploy', 'cmd_deploy', 'cloud', 'Cloud model deployment')(args)

def lazy_cmd_test(args):
    """Lazy loader for test command."""
    return lazy_import_command('namel3ss.cli.commands.test', 'cmd_test', 'cli', 'Testing framework')(args)

def lazy_cmd_lint(args):
    """Lazy loader for lint command."""
    return lazy_import_command('namel3ss.cli.commands.lint', 'cmd_lint', 'cli', 'Code linting')(args)

def lazy_cmd_typecheck(args):
    """Lazy loader for typecheck command."""
    return lazy_import_command('namel3ss.cli.commands.typecheck', 'cmd_typecheck', 'cli', 'Type checking')(args)

def lazy_cmd_format(args):
    """Lazy loader for format command."""
    return lazy_import_command('namel3ss.cli.commands.format', 'cmd_format', 'cli', 'Code formatting')(args)

def lazy_cmd_lsp(args):
    """Lazy loader for LSP command."""
    return lazy_import_command('namel3ss.cli.commands.lsp', 'cmd_lsp', 'cli', 'Language server protocol')(args)