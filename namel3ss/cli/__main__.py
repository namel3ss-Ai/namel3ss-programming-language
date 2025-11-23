"""
Main entry point for the Namel3ss CLI when run as a module.

This allows the CLI to be executed using:
    python -m namel3ss.cli

or the equivalent console script entry point.
"""

from . import main

if __name__ == '__main__':
    main()