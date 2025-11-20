"""
Doctor command implementation.

This module handles the 'doctor' subcommand for checking system health
and dependency availability.
"""

import argparse
import sys

from namel3ss.utils.dependencies import iter_dependency_reports

from ..errors import handle_cli_exception


def cmd_doctor(args: argparse.Namespace) -> None:
    """
    Handle the 'doctor' subcommand to check dependency availability.
    
    Reports the status of core and optional dependencies, checking if
    required packages are installed and providing installation advice
    for missing packages.
    
    Args:
        args: Parsed command-line arguments (no specific args required)
    
    Raises:
        SystemExit: If any core dependencies are missing
    
    Examples:
        >>> args = argparse.Namespace()
        >>> cmd_doctor(args)  # doctest: +SKIP
        ✓ Core Python: available
        ✓ Core Parser: available
        ✗ Optional Uvicorn: missing
            Missing modules: uvicorn
            → Install with: pip install uvicorn[standard]
    """
    try:
        reports = iter_dependency_reports()
        core_missing = False
        
        for report in reports:
            prefix = "Optional" if report.optional else "Core"
            status = "available" if not report.missing else "missing"
            symbol = "✓" if status == "available" else "✗"
            print(f"{symbol} {prefix} {report.title}: {status}")
            
            if report.missing:
                missing_modules = ", ".join(report.missing)
                print(f"    Missing modules: {missing_modules}")
                if report.advice:
                    print(f"    → {report.advice}")
                if not report.optional:
                    core_missing = True
        
        if core_missing:
            sys.exit(1)
    
    except Exception as exc:
        handle_cli_exception(exc)
