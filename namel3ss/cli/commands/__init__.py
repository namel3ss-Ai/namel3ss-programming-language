"""
CLI command modules.

This package contains individual command implementations for the Namel3ss CLI.
Each command module handles a specific CLI subcommand (build, run, eval, etc.).
"""

from .build import cmd_build
from .deploy import cmd_deploy
from .doctor import cmd_doctor
from .eval import cmd_eval, cmd_eval_suite
from .run import cmd_run
from .tools import cmd_format, cmd_lint, cmd_lsp, cmd_test, cmd_typecheck
from .train import cmd_train
from .debug import cmd_debug, add_debug_command
from .cmd_stdlib import cmd_stdlib, add_stdlib_command
from .packages import cmd_packages, add_packages_command
from .modules import cmd_modules, add_modules_command

__all__ = [
    "cmd_build",
    "cmd_deploy",
    "cmd_doctor",
    "cmd_eval",
    "cmd_eval_suite",
    "cmd_format",
    "cmd_lint",
    "cmd_lsp",
    "cmd_run",
    "cmd_test",
    "cmd_train",
    "cmd_typecheck",
    "cmd_debug",
    "add_debug_command",
    "cmd_stdlib",
    "add_stdlib_command",
    "cmd_packages",
    "add_packages_command",
    "cmd_modules", 
    "add_modules_command",
]
