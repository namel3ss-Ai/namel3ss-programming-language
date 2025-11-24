"""
CLI command for running Namel3ss Language conformance tests.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from namel3ss.conformance import ConformanceRunner, TestResult


def cmd_conformance(args: argparse.Namespace, ctx: Optional[object] = None) -> int:
    """
    Run conformance tests for Namel3ss Language 1.0.
    
    Args:
        args: Parsed command-line arguments
        ctx: Optional CLI context
    
    Returns:
        Exit code (0 if all tests pass, non-zero otherwise)
    """
    # Determine conformance test directory
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        # Default to tests/conformance/v1 relative to repo root
        # Try to find it relative to this file
        cli_file = Path(__file__).resolve()
        repo_root = cli_file.parent.parent.parent.parent
        test_dir = repo_root / "tests" / "conformance" / "v1"
    
    if not test_dir.exists():
        print(f"Error: Conformance test directory not found: {test_dir}", file=sys.stderr)
        return 1
    
    # Create runner
    runner = ConformanceRunner(verbose=args.verbose)
    
    # Run tests
    print(f"Running conformance tests from: {test_dir}")
    if args.category:
        print(f"Category filter: {args.category}")
    if args.test:
        print(f"Test filter: {args.test}")
    print()
    
    try:
        results = runner.run_all_tests(
            test_dir=test_dir,
            category=args.category,
            test_id=args.test
        )
    except Exception as e:
        print(f"Error running conformance tests: {e}", file=sys.stderr)
        return 1
    
    # Output results
    if args.format == "json":
        # Machine-readable JSON output
        output = {
            "conformance_version": "1.0.0",
            "language_version": "1.0.0",
            "test_directory": str(test_dir),
            "filters": {
                "category": args.category,
                "test_id": args.test
            },
            "results": [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        if not args.verbose:
            # Print summary for each test
            for result in results:
                symbol = {
                    TestResult.PASS: "✓",
                    TestResult.FAIL: "✗",
                    TestResult.SKIP: "○",
                    TestResult.ERROR: "E"
                }[result.result]
                
                status = result.result.value.upper()
                print(f"{symbol} [{status:5}] {result.test_id:30} {result.test_name}")
        
        # Print summary
        runner.print_summary()
    
    # Exit with appropriate code
    failed = sum(1 for r in results if r.result == TestResult.FAIL)
    errors = sum(1 for r in results if r.result == TestResult.ERROR)
    
    if failed > 0 or errors > 0:
        return 1
    
    return 0


def add_conformance_command(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'conformance' command to the CLI.
    
    Args:
        subparsers: Subparser action to add the command to
    """
    conformance_parser = subparsers.add_parser(
        "conformance",
        help="Run language conformance tests",
        description="""
        Run the Namel3ss Language 1.0 conformance test suite.
        
        The conformance suite validates language-level behavior and can be used by
        multiple implementations to verify correctness. Tests cover parsing, type
        checking, and runtime semantics.
        
        Examples:
            namel3ss conformance                          # Run all tests
            namel3ss conformance --category parse         # Run only parse tests
            namel3ss conformance --test parse-valid-001   # Run specific test
            namel3ss conformance --verbose                # Show detailed output
            namel3ss conformance --format json            # Machine-readable output
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    conformance_parser.add_argument(
        "--test-dir",
        metavar="PATH",
        help="Path to conformance test directory (default: tests/conformance/v1)"
    )
    
    conformance_parser.add_argument(
        "--category",
        choices=["parse", "types", "runtime"],
        help="Run only tests in specified category"
    )
    
    conformance_parser.add_argument(
        "--test",
        metavar="TEST_ID",
        help="Run specific test by ID (e.g., parse-valid-001)"
    )
    
    conformance_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed test output"
    )
    
    conformance_parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)"
    )
    
    conformance_parser.set_defaults(func=cmd_conformance)


__all__ = ["cmd_conformance", "add_conformance_command"]
