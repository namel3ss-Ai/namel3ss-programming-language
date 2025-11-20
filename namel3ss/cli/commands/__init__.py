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
from .tools import cmd_lint, cmd_lsp, cmd_test, cmd_typecheck
from .train import cmd_train

__all__ = [
    "cmd_build",
    "cmd_deploy",
    "cmd_doctor",
    "cmd_eval",
    "cmd_eval_suite",
    "cmd_lint",
    "cmd_lsp",
    "cmd_run",
    "cmd_test",
    "cmd_train",
    "cmd_typecheck",
]
