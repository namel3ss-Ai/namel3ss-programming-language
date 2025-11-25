"""
CLI command modules.

This package contains individual command implementations for the Namel3ss CLI.
Each command module handles a specific CLI subcommand (build, run, eval, etc.).

Commands are now lazy-loaded to avoid requiring optional dependencies for core functionality.
"""

# Import only core commands that work with minimal dependencies
from .build import cmd_build
from .run import cmd_run
from .doctor import cmd_doctor

# Lazy imports for optional commands
def _lazy_import_deploy():
    from .deploy import cmd_deploy
    return cmd_deploy

def _lazy_import_debug():
    try:
        from .debug import cmd_debug, add_debug_command
        return cmd_debug, add_debug_command
    except ImportError:
        def missing_debug(*args, **kwargs):
            print("Error: Debug features require the 'cli' extra.")
            print("Install with: pip install 'namel3ss[cli]'")
            import sys
            sys.exit(1)
        return missing_debug, missing_debug

def _lazy_import_local_deploy():
    try:
        from .local_deploy import cmd_deploy_local, add_local_deploy_command
        return cmd_deploy_local, add_local_deploy_command
    except ImportError:
        def missing_local_deploy(*args, **kwargs):
            print("Error: Local deployment features require the 'local-deploy' extra.")
            print("Install with: pip install 'namel3ss[local-deploy]'")
            import sys
            sys.exit(1)
        return missing_local_deploy, missing_local_deploy

def _lazy_import_ai_commands():
    try:
        from .eval import cmd_eval, cmd_eval_suite
        from .train import cmd_train
        return cmd_eval, cmd_eval_suite, cmd_train
    except ImportError:
        def missing_ai(*args, **kwargs):
            print("Error: AI features require the 'ai' extra.")
            print("Install with: pip install 'namel3ss[ai]'")
            import sys
            sys.exit(1)
        return missing_ai, missing_ai, missing_ai

def _lazy_import_dev_tools():
    try:
        from .tools import cmd_format, cmd_lint, cmd_lsp, cmd_test, cmd_typecheck
        return cmd_format, cmd_lint, cmd_lsp, cmd_test, cmd_typecheck
    except ImportError:
        def missing_dev(*args, **kwargs):
            print("Error: Development tools require the 'dev' extra.")
            print("Install with: pip install 'namel3ss[dev]'")
            import sys
            sys.exit(1)
        return missing_dev, missing_dev, missing_dev, missing_dev, missing_dev

def _lazy_import_enhanced_cli():
    try:
        from .cmd_stdlib import cmd_stdlib, add_stdlib_command
        from .packages import cmd_packages, add_packages_command
        from .modules import cmd_modules, add_modules_command
        from .security import add_security_command
        from .conformance import cmd_conformance, add_conformance_command
        return (cmd_stdlib, add_stdlib_command, cmd_packages, add_packages_command,
                cmd_modules, add_modules_command, add_security_command,
                cmd_conformance, add_conformance_command)
    except ImportError:
        def missing_cli(*args, **kwargs):
            print("Error: Enhanced CLI features require the 'cli' extra.")
            print("Install with: pip install 'namel3ss[cli]'")
            import sys
            sys.exit(1)
        return (missing_cli,) * 8

# Export core commands immediately
__all__ = [
    "cmd_build",
    "cmd_run", 
    "cmd_doctor",
    # Lazy loaders
    "_lazy_import_deploy",
    "_lazy_import_debug",
    "_lazy_import_local_deploy", 
    "_lazy_import_ai_commands",
    "_lazy_import_dev_tools",
    "_lazy_import_enhanced_cli",
]
